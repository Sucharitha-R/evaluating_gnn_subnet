## Evaluating the Explainability of Graph Neural Networks for Disease Subnetwork Detection

This study was carried out as a final thesis project for the Bachelor Computer Science and Engineering, TU Delft. 

The full paper is available [here](http://resolver.tudelft.nl/uuid:6b1f4281-3b9f-44b1-8cf0-61d0a4706635).

This repository extends GNN-SubNet [1] and adds four explainability evaluation metrics:
- RDT-fidelity (faithfulness)
- Sparsity
- Validity+
- Validity-

RDT-fidelity and Sparsity are metrics defined by the BAGEL benchmark [2], while validity+ and validity- are newly defined by this study taking inspiration from [3] and [4].

## GNN-SubNet

The paper describing the original GNN-SubNet project is available here: <https://academic.oup.com/bioinformatics/article/38/Supplement_2/ii120/6702000> 

The code here builds upon the existing GitHub repository (containing both code and data) of GNN-SubNet: <https://github.com/pievos101/GNN-SubNet/tree/main>.

## Installation

First, set up a python environment and install the following packages:

```python
pip3 install torch torchvision torchaudio
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install pandas
pip install python-igraph
pip install dgl
pip install matplotlib

pip install GNNSubNet
```

## Usage

```explainability_evaluator.py``` implements four evaluation metrics: RDT-fidelity, validity+, validity- and sparsity. Check this file for detailed documentation on each of the methods.

The snippet below is intended to give an idea of how this evaluation is done. For a full working example, please refer to ```evaluation_experiments.ipynb```.

```
g = gnn.GNNSubNet(loc, ppi, feats, targ, normalize=False)
g.train()
g.explain(10) # 10 runs
ev = eval.explainability_evaluator(g)
ev.evaluate_RDT_fidelity(use_softmask=True, samples = samples)
ev.evaluate_sparsity()
ev.evaluate_validity(threshold=t, confusion_matrix=True)
```

## Jupyter notebooks

Three notebooks can be found in this repository:
- ```evaluation_experiments.ipynb``` : a full working example, demonstrating how to train, explain and evaluate the explanations using the four explainability metrics.
- ```visualisation_and_analyis.ipynb```: intended as a follow-up to the first notebook, this takes the results of the experiments and creates processed tables and plots. These were used to generate the tables and plots presented in the [paper](http://resolver.tudelft.nl/uuid:6b1f4281-3b9f-44b1-8cf0-61d0a4706635).
- ```extended_experiments.ipynb```: an additional experiment that looks into the size of the disease subnetworks found after training and explaining the GNN.
  
## References
[1] B. Pfeifer, A. Secic, A. Saranti, and A. Holzinger, “Gnn-
subnet: Disease subnetwork detection with explainable
graph neural networks,” Jan. 2022. DOI: [10.1101/2022.
01.12.475995.](https://academic.oup.com/bioinformatics/article/38/Supplement_2/ii120/6702000)

[2] M. Rathee, T. Funke, A. Anand, and M. Khosla, “Bagel:
A benchmark for assessing graph neural network expla-
nations,” arXiv preprint [arXiv:2206.13983, 2022.](https://arxiv.org/abs/2206.13983)

[3] H. Yuan, H. Yu, S. Gui, and S. Ji, Explainability in
graph neural networks: A taxonomic survey, 2022.
[arXiv: 2012.15445 [cs.LG].](https://arxiv.org/abs/2012.15445)

[4] T. Funke, M. Khosla, M. Rathee, and A. Anand, Zorro:
Valid, sparse, and stable explanations in graph neural
networks, 2022. [arXiv: 2105.08621 [cs.LG].](https://arxiv.org/abs/2105.08621)
