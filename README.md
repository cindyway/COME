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

<!--- The paper's utilized dataset is available in h5ad files on [Dropbox](https://www.dropbox.com/scl/fi/3ywqslcj18kflgppnwwhq/data.zip?rlkey=aghlshs3mos97g94j8gl443zm&st=ocok6ab1&dl=0). --->
Our input data is in the standard format of [anndata format](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html), stored as h5ad files. 
All the data can be obtained from the references cited in the paper. 

## Model Training
To replicate the COME model, please execute the train_eval.py script.

For instance: 
```bash
python train_val.py --pretrain --train
```
Adjustable (hyper)parameters can be customized in the configure.py file.


## Examples

To use the output mapping for gene reconstruction and spot deconvolution, please take a look at the [Tutorial.ipynb](https://github.com/cindyway/COME/blob/main/Tutorial.ipynb) script.

## Acknowledgement
Some implementations of metrics and the visualization in this Repo are based on [SpatialBenchmarking](https://github.com/QuKunLab/SpatialBenchmarking) and [GraphST](https://github.com/JinmiaoChenLab/GraphST)


