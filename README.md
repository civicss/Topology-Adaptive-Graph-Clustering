# TAGC: Topology-Adaptive Graph Clustering via Distribution-Guided Attention

This repository contains the official PyTorch implementation of the paper **"TAGC: Topology-Adaptive Graph Clustering via Distribution-Guided Attention"**. 

TAGC is a novel graph clustering framework designed to bridge the semantic gap between localized structural encoding and macro-level clustering. By establishing a bidirectional closed-loop synergy, TAGC dynamically filters and aligns multi-scale higher-order topological views under the guidance of global cluster distributions, effectively preventing representation collapse.

## Requirements

The codebase is implemented in Python and requires the following key packages:

* Python 3.9+
* PyTorch
* NumPy
* SciPy
* scikit-learn

## Datasets

We evaluate our model on six widely used benchmark datasets, which include graph datasets, image datasets, and text/record datasets. 

Due to file size limits, the raw datasets are not included in this repository. You can download the public datasets from their original sources and place the feature matrices (`{name}.txt`) and label files (`{name}_label.txt`) into the `data/` directory.

**Graph Datasets:**

* **ACM & CiteSeer**: [ACM Digital Library](http://dl.acm.org) & [CiteSeerX](http://CiteSeerx.ist.psu.edu/)
* **DBLP**: [DBLP Computer Science Bibliography](https://dblp.uni-trier.de)

**Non-Graph Datasets:**

* **USPS**: Image dataset for handwritten text recognition research (Hull, 1994).
* **Reuters**: Text dataset for categorization research (Lewis et al., 2004).
* **HHAR**: Record dataset for heterogeneous activity recognition (Stisen et al., 2015).

## Quick Start

The complete pipeline consists of graph construction, autoencoder pre-training, and end-to-end TAGC training. We use ACM as an example below.

### Step 1: Graph Construction

For non-graph datasets (e.g., USPS, HHAR, Reuters), first construct the KNN graph:

```bash
python KNNGraph.py --name usps --k 3
```

For all datasets, generate the higher-order similarity enhanced graph and the Motif-augmented graphs:

Bash

```
# Generate second-order similarity enhanced graph
python graph_enhance.py 

# Generate Triangle, Quadrangle, and Pentagonal motif graphs
python motif_generate.py 
```

*Note: Make sure to modify the `dataset` name inside the scripts accordingly.*

### Step 2: Autoencoder Pre-training

Pre-train the Autoencoder (AE) to initialize the node attribute embeddings. The pre-trained weights will be saved in `data/pkl/`.

Bash

```
python pretrain.py --name acm --n_input 1870 --n_clusters 3 --epoch 30
```

### Step 3: End-to-End TAGC Training

Run the main training script. The model will automatically load the pre-trained AE weights, original graphs, and motif graphs to perform topology-adaptive clustering.

Bash

```
python train_acm.py --name acm
```

The clustering metrics (ACC, NMI, ARI, F1) will be printed in the console during training, and the best results will be recorded.
