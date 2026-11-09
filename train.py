"""Fit the full-data smFISH mapping consumed by tutorial.ipynb."""

from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from scipy.sparse import issparse

from configure import get_default_config
from datasets import load_data
from model import Model
from train_eval import device, pretrain_ae, train
from utils import normalize_type


def main():
    np.random.seed(48)
    torch.manual_seed(48)

    rna, spot = load_data('smFISH')
    for adata in (rna, spot):
        adata.var_names_make_unique()
        if issparse(adata.X):
            adata.X = adata.X.toarray()
    spot, rna = normalize_type(spot, rna)
    x1 = np.asarray(spot.X, dtype=np.float32)
    x2 = np.asarray(rna.X, dtype=np.float32)
    labels = rna.obs['cell_type'].to_numpy()
    type_mask = (labels[:, None] == labels[None, :]).astype(np.float32)

    config = get_default_config('smFISH')
    config.update(dims=[x1.shape[1], *config['dims'][1:]],
                  num_sample1=x1.shape[0], num_sample2=x2.shape[0])
    model = Model(config).to(device)
    for directory in ('pretrain', 'result', 'coef'):
        Path(directory).mkdir(exist_ok=True)
    pretrain_ae(model.ae, torch.optim.Adam(model.ae.parameters(), lr=config['pre_lr']),
                np.concatenate((x1, x2)), config, 'pretrain/smFISH_full_ae.pkl')
    train(model, torch.optim.Adam(model.parameters(), lr=config['lr']),
          x1, x2, type_mask, config, 'result/smFISH_full_model.pkl')
    savemat('coef/smFISH_C.mat', {'C': model.map.Coefficient.detach().cpu().numpy()})
    print('Wrote coef/smFISH_C.mat for tutorial2026.IPYNB')


if __name__ == '__main__':
    main()
