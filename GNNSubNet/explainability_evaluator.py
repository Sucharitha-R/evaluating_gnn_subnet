# Evaluating the explainability of GNNSubNet
# Authored by Sucharitha Rajesh

import numpy as np
from scipy.stats import entropy
import torch
from .gnn_training_utils import pass_data_iteratively
from .dataset import convert_to_s2vgraph
from torch_geometric.data.data import Data
from .graph_dataset import GraphDataset

class explainability_evaluator():

    def __init__(self, trained_gnnsubnet) -> None:

        # Copy the parameters from the trained GNNSubNet object that are needed for the evaluation:
        # datasets, node masks (explanations), and the GNN model itself.

        self.dataset = trained_gnnsubnet.dataset
        self.test_dataset = trained_gnnsubnet.test_dataset
        self.node_mask = trained_gnnsubnet.node_mask
        self.node_mask_matrix = trained_gnnsubnet.node_mask_matrix
        self.model = trained_gnnsubnet.model

    def evaluate_RDT_fidelity(self, use_softmask=True, samples=10, threshold=30, device="cpu"):
        """
        Evaluates RDT-fidelity using random perturbations of each graph in the test dataset.

        :param use_softmask: whether to use a soft mask (the existing node mask in GNNSubNet) or to transform it into a hard mask
        :param samples: the number of perturbations done for each graph, default = 10
        :param threshold: the percentage of nodes to select in a hard mask. Only applicable when use_softmask = False. 
        Ex. if threshold = 30, the top 30% of nodes from the soft mask are set to important in the hard mask.

        :return: a list of fidelity values, one for each graph in the test dataset.
        """
        fidelities = []
        print(f"Evaluating RDT-fidelity using {len(self.test_dataset)} graphs from the test dataset")

        if(use_softmask):
            print("Using a soft mask for evaluating RDT-fidelity")
            node_mask = self.node_mask
        else:
            print(f"Using a hard mask for evaluating RDT-fidelity, with thresold {threshold}")
            # Convert soft to hard mask
            cutoff = np.percentile(self.node_mask, 100 - threshold)
            node_mask = np.greater(self.node_mask, cutoff).astype(np.float32)
        
        # Find the global distribution of the data
        (num_nodes, num_features) = self.dataset[0].x.size()
        global_data = torch.empty(size=(len(self.dataset), num_nodes, num_features))
        for (i, input_graph) in enumerate(self.dataset):
            global_data[i] = input_graph.x
        
        # For each test graph, compute fidelity over given number of samples
        c = 0
        for input_graph in self.test_dataset:
            c = c+1
            print(f"Evaluating {c} out of {len(self.test_dataset)}")
            feature_matrix = input_graph.x

            # Set up the node/feature mask
            feature_mask = torch.ones((1,num_features), device=device)
            node_mask = torch.tensor(node_mask).view(1, num_nodes).to(device)
            mask = node_mask.T.matmul(feature_mask)
            
            # Find initial predicted label
            predicted_class = self.predict_helper(input_graph)

            # Generate random numbers to serve as indices on the global data
            random_indices = torch.randint(len(self.dataset), (samples, ))

            # Select the required number of random feature matrices from the global data
            random_feature_matrices = torch.index_select(global_data, 0, random_indices)
            
            correct = 0.0
            for i in range(samples):
                random_feature_matrix = random_feature_matrices[i, :, :]
                randomized_features = mask * feature_matrix + (1 - mask) * random_feature_matrix
                
                # Find the prediction on the perturbed graph
                distorted_class = self.predict_helper(input_graph, perturbed_features=randomized_features)
                
                # Compare the prediction of the input graph with the perturbed graph
                if distorted_class == predicted_class:
                    correct += 1
            fidelities.append(correct / samples)

        return fidelities


    def evaluate_sparsity(self):
        """
        Finds the sparsity score of the node masks found by each run of the explainer.

        :return: A list of sparsity values, one for each node mask.
        Ex. If the explainer was done with 10 runs, 10 masks are stored in node_mask_matrix, 
        this method returns a list of 10 values giving the sparsity of each mask.
        """
        sparsities = []
        node_masks = self.node_mask_matrix
        num_masks = node_masks.shape[1]
        print(f"Evaluating sparsity on {num_masks} node masks")
        # Calculate minimum possible entropy
        num_nodes = node_masks.shape[0]
        max_entropy = entropy( np.ones(num_nodes) / num_nodes )

        for node_mask in node_masks.T:
            # Normalise
            node_mask = node_mask / np.sum(node_mask)
            # Compute entropy
            e = entropy(node_mask)
            # Convert entropy to sparsity score
            score = 1 -  (e / max_entropy)
            sparsities.append(score)
        return sparsities


    def evaluate_validity(self, threshold=30, device='cpu', confusion_matrix=True):
        """
        Validity+ and Validity- evaluated on the test dataset.
        Uses the stored node mask to set important / unimportant features to average values
        and check whether the predicted label stays the same.

        :param threshold: the percentage of nodes to select in the hard mask. 
        Ex. if threshold = 30, the top 30% of nodes from the soft mask are set to important in the hard mask.
        :param confusion_matrx: whether or not to compute and return the confusion matrices (original class vs perturbed class)

        :return: validity+ score, validity- score, (optional) validity+ confusion matrix, (optional) validity- confusion matrix
        """

        print(f"Evaluating Validity+ and Validity- using {len(self.test_dataset)} graphs from the test dataset")

        # Set up confusion matrices
        if confusion_matrix:
            plus_conf_matrix = np.zeros((2,2))
            minus_conf_matrix = np.zeros((2,2))

        # Find average values over the test dataset 
        dataset = self.test_dataset
        global_average = self.find_node_averages().to(device)

        # Convert soft to hard mask
        cutoff = np.percentile(self.node_mask, 100 - threshold)
        hard_node_mask = np.greater(self.node_mask, cutoff).astype(np.float32)

        # Iterate over the test graphs
        mask = self.node_mask
        validity_plus = 0
        validity_minus = 0
        c = 0
        for input_graph in dataset:
            c = c+1
            print(f"Evaluating {c} out of {len(dataset)}")
            full_feature_matrix = input_graph.x.to(device)
            (num_nodes, num_features) = full_feature_matrix.size()

            # Set up the node/feature mask
            feature_mask = torch.ones((1,num_features), device=device)
            node_mask = torch.tensor(hard_node_mask).view(1, num_nodes).to(device)
            mask = node_mask.T.matmul(feature_mask)

            # Find initial predicted label
            predicted_class = self.predict_helper(input_graph)

            # Validity- : keep all the important features, average unimportant features
            # If the prediction stays the same, it is better.
            averaged_features_minus = mask * full_feature_matrix + (1 - mask) * global_average
            distorted_class = self.predict_helper(input_graph, perturbed_features=averaged_features_minus)
            if distorted_class == predicted_class:
                validity_minus += 1
            if confusion_matrix:
                minus_conf_matrix[predicted_class][distorted_class] += 1

            # Validity+ : keep all the unimportant features, average important features. 
            # If the prediction changes, it is better.
            averaged_features_plus = mask * global_average + (1 - mask) * full_feature_matrix
            distorted_class = self.predict_helper(input_graph, perturbed_features=averaged_features_plus)
            if distorted_class != predicted_class:
                validity_plus += 1
            if confusion_matrix:
                plus_conf_matrix[predicted_class][distorted_class] += 1

        if confusion_matrix:
                return validity_plus/len(dataset), validity_minus/len(dataset), plus_conf_matrix, minus_conf_matrix
        else:
            return validity_plus/len(dataset), validity_minus/len(dataset)


    def find_node_averages(self):
        """
        Helper method for evaluation:
        finds and returns the global average of the feature values of each node
        in the form of a tensor with dimensions (no. of nodes) * (no. of features)
        """
        dataset = self.test_dataset
        (num_nodes, num_features) = dataset[0].x.size()

        # Construct a single tensor with the feature matrices of all graphs
        full = torch.empty(size=(len(dataset), num_nodes, num_features))
        for i in range(len(dataset)):
            full[i] = dataset[i].x
        
        # Take the average along all graphs
        return torch.mean(full, 0)


    def predict_helper(self, input_graph, perturbed_features=None):
        """
        Helper method for evaluation:
        finds the model's predicted class on the given input graph
        or (if provided) on the input graph with perturbed features
        """
        if perturbed_features is not None:
            # GNNSubNet has the model tensors on the CPU by default
            test_graph = Data(x=torch.tensor(perturbed_features).float().to("cpu"),
                                    edge_index = input_graph.edge_index,
                                    y = input_graph.y
                            )
        else:
            test_graph = input_graph
        s2v_test_graph  = convert_to_s2vgraph(GraphDataset([test_graph]))
        model = self.model
        model.eval()
        output = pass_data_iteratively(model, s2v_test_graph)
        predicted_class = output.max(1, keepdim=True)[1]
        return predicted_class