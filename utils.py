# -*- coding:utf-8 _*-
__author__ = 'xindiawei2'
__date__ = '6/9/2023 4:10 pm'
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
import torch
import scipy.io as scio
import pandas as pd


class EarlyStopper:
    def __init__(self, patience=3, min_delta=10):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def filter_with_overlap_gene(adata_st, adata_sc):
    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata_st.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))


    st = adata_st[:, genes]
    sc = adata_sc[:, genes]

    return st, sc

def normalize_type(x1,x2,type='image'):
    spot = x1.copy()
    rna = x2.copy()

    if type == 'image':
        train_scaler = MinMaxScaler()
        spot.X = train_scaler.fit_transform(spot.X)

        test_scaler = MinMaxScaler()
        rna.X = test_scaler.fit_transform(rna.X)
    elif type == 'sequence':
        hvg = 3000

        sc.pp.normalize_total(spot, target_sum=1e4)
        sc.pp.log1p(spot)


        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        #sc.pp.highly_variable_genes(rna, n_top_genes=hvg, inplace=True, subset=True)

    spot, rna = filter_with_overlap_gene(spot, rna)


    return spot, rna


def calculate_cosine_similarity(matrix1, matrix2):

    norm_matrix1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
    norm_matrix2 = np.linalg.norm(matrix2, axis=1, keepdims=True)


    dot_product = np.dot(matrix1, matrix2.T)


    cosine_sim = dot_product / (norm_matrix1 * norm_matrix2.T)

    return cosine_sim
def cell_type_2_martix(rna,data_name):
    if 'cell_type' in rna.obs:
        cell_type_list = rna.obs['cell_type']
    else:
        print("No cell type information")
        cell_type_list = rna.obs_names.tolist()


    matrix = np.array(cell_type_list)
    matrix = matrix.reshape((len(cell_type_list), 1))

    mask_matrix = (matrix == matrix.T).astype(np.float32)

    # Save the mask matrix to a .mat file
    scio.savemat(f"data/{data_name}_type_mask.mat", {'Mask': mask_matrix})
    return mask_matrix
def cal_ssim(im1, im2, M=None):
    """
    Calculate the SSIM value between two PyTorch tensors.

    Parameters:
    - im1: Tensor, shape dimension = 2
    - im2: Tensor, shape dimension = 2
    - M: the max value in [im1, im2]

    Returns:
    - ssim: SSIM value
    """

    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    M = [im1.max(), im2.max()][im1.max() > im2.max()]

    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = torch.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = torch.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()

    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2

    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    # print (l12, c12, s12)
    ssim = l12 * c12 * s12

    return ssim

def project_cell_to_spot(adata, adata_sc, retain_percent=0.1):
    '''\
    Project cell types onto ST data using mapped matrix in adata.obsm

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    adata_sc : anndata
        AnnData object of scRNA-seq reference data.
    retrain_percent: float
        The percentage of cells to retain. The default is 0.1.
    Returns
    -------
    None.

    '''

    # read map matrix
    map_matrix = adata.obsm['map_matrix']   # spot x cell

    # extract top-k values for each spot
    map_matrix = extract_top_value(map_matrix) # filtering by spot

    # construct cell type matrix
    matrix_cell_type = construct_cell_type_matrix(adata_sc)
    matrix_cell_type = matrix_cell_type.values

    # projection by spot-level
    matrix_projection = map_matrix.dot(matrix_cell_type)

    # rename cell types
    cell_type = list(adata_sc.obs['cell_type'].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    # cell_type = [s.replace(' ', '_') for s in cell_type]
    df_projection = pd.DataFrame(matrix_projection, index=adata.obs_names, columns=cell_type)  # spot x cell type

    # normalize by row (spot)
    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)

    # add projection results to adata
    adata.obs[df_projection.columns] = df_projection

    return df_projection
def extract_top_value(map_matrix, retain_percent=0.1):
    '''\
    Filter out cells with low mapping probability

    Parameters
    ----------
    map_matrix : array
        Mapped matrix with m spots and n cells.
    retain_percent : float, optional
        The percentage of cells to retain. The default is 0.1.

    Returns
    -------
    output : array
        Filtered mapped matrix.

    '''

    # retain top 1% values for each spot
    top_k = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)

    return output
def construct_cell_type_matrix(adata_sc):
    label = 'cell_type'
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    #res = mat.sum()
    return mat
