# COME

This source code repository aims to reproduce the approach introduced in the paper "COME: contrastive mapping learning for spatial reconstruction of scRNA-seq data."

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

```bash
conda create python=3.7 -n COME
conda activate COME
```
Make sure to install the corresponding PyTorch CUDA version by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/)

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

To use the output mapping for gene reconstruction and spot deconvolution, please take a look at the [Tutorial.py](https://github.com/cindyway/COME/blob/main/Tutorial.ipynb) script.

## Acknowledgement
Some implementations of metrics and the visualization in this Repo are based on [SpatialBenchmarking](https://github.com/QuKunLab/SpatialBenchmarking) and [GraphST](https://github.com/JinmiaoChenLab/GraphST)


