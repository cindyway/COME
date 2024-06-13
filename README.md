# COME

This source code repository aims to reproduce the approach introduced in the paper titled "COME: contrastive mapping learning for spatial reconstruction of scRNA-seq data."

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Model Training](#modeltraining)
- [Examples](#examples)


## Installation

The installation requirements and dependencies for the project. For example:

- Python 3.7+
- PyTorch 1.6+
- Other dependencies

You can also provide installation instructions like:

```bash
pip install -r requirements.txt
```

## Data

The dataset utilized in the paper is provided in an h5ad file for convenient storage and access. The raw data can be downloaded from the references as cited in the paper.

## Model Training
To replicate the COME model, please execute the train_eval.py script.

For instance: 
```bash
python train_val.py --pretrain --train
```
Adjustable (hyper)parameters can be customized in the configure.py file.


## Examples

For utilizing the output mapping for gene reconstruction and spot deconvolution, please refer to the Tutorial.py script.


