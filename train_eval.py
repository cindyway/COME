
import numpy as np
import torch
import argparse
import scipy.io as sio

from torch.utils.data import DataLoader
from configure import get_default_config
from datasets import load_data, Data_Sampler, TrainDataset
from model import Model
from utils import EarlyStopper, normalize_type,cell_type_2_martix
import pandas as pd
import random

from sklearn.model_selection import KFold
from GenesMetrics import count

import datetime
import os


parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=int, default=1  )
parser.add_argument('--sttype', type=str, default='image')
parser.add_argument('--pretrain', action='store_false')
parser.add_argument('--train', action='store_false')
parser.add_argument('--patience', type=int, default=10)

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = {
    0: "dro",
    1: "smFISH",
    2: "MERFISH",
    3: "PDAC",
    4: "STARmap",
}
data_name = dataset[args.dataset]

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
config = get_default_config(data_name)
pre_name = '_'.join(str(dim) for dim in config['dims'][1:])

def main():

    random.seed(48)
    ktimes = 10
    cv = KFold(n_splits=ktimes, shuffle=False)

    # load datasets
    rna, spot = load_data(data_name)
    rna.var_names_make_unique()
    spot.var_names_make_unique()
    spot_n, rna_n = normalize_type(spot, rna, type=args.sttype)
    atlas_genes = spot_n.var_names.values
    overlap_list=atlas_genes
    print('Overlap Genes:', len(overlap_list))
    type_mask_path = "data/{}_type_mask.mat".format(data_name)
    if os.path.exists(type_mask_path):
        type_mask = sio.loadmat(type_mask_path)['Mask']
    else:
        type_mask = cell_type_2_martix(rna_n,  data_name=data_name)

    x1_cell = spot_n.copy()  # ref
    x2_rna = rna_n.copy()
    cnts = 0
    tmp_dims = 0
    vis_pcc = []
    result = pd.DataFrame()
    for train_idx, test_idx in cv.split(overlap_list):
        cnts += 1
        train_gene = [overlap_list[i] for i in train_idx]
        predict_gene = [overlap_list[i] for i in test_idx]

        x1_train_cell = x1_cell[:, train_gene].X
        x2_train_cell = x2_rna[:, train_gene].X

        x1 = x1_train_cell.astype(np.float32)
        x2 = x2_train_cell.astype(np.float32)

        config = get_default_config(data_name)
        tool = data_name + str(config['epochs'])
        outdir = 'eval_table/'+ data_name +"/"

        config['num_sample1'] = x1.shape[0]
        config['num_sample2'] = x2.shape[0]
        # model initialize
        pre_name = '%s_%s_%s' % (x1.shape[1], config['dims'][1], config['dims'][2])
        config['dims'][0] = x1.shape[1]
        pretrain_path = "pretrain/%s_%s.pkl" % (data_name, pre_name)
        print (pretrain_path, tmp_dims, x1.shape[1])
        if tmp_dims != x1.shape[1] and cnts > 0:
            if_pretrain = True
        elif cnts == 0:
            if_pretrain = True
        else:
            if_pretrain = False
        tmp_dims = x1.shape[1]
        model = Model(config)
        model.to(device)
        optimizer_pre = torch.optim.Adam(model.ae.parameters(), lr=config['pre_lr'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        print (args.pretrain)
        if args.pretrain:
            pretrain_ae(model.ae, optimizer_pre, np.concatenate((x1, x2), axis=0), config, pretrain_path)
        model.ae.load_state_dict(torch.load(pretrain_path))
        model_path = "result/%s_model.pkl" % (data_name)
        model.train()
        if args.train:
            train(model, optimizer, x1, x2,type_mask,config, model_path)
        model.eval()
        # eval metric
        print ('testing...')
        rna_test = rna_n[:, predict_gene].X
        atlas = spot_n.to_df()
        atlas.reset_index(drop=True, inplace=True)
        C_after_softmax = model.map.Coefficient.detach().cpu().numpy()
        hat_atlas = predict_result(rna_test, C_after_softmax, predict_gene)
        result = pd.concat([result, hat_atlas], axis=1)
        atlas = atlas.loc[:, predict_gene]
        metric=[]
        results = count(atlas, hat_atlas, tool, outdir, metric)
        resultsall, tmp_pcc = results.compute_all(cnts)
        vis_pcc.append(tmp_pcc)
        print("Finished " + str(cnts) + ' Times Training and Testing...')

    result.to_csv(outdir  + data_name+"_impute.csv", header=1, index=1)

def predict_result(rna,OT,genes_names):
    spot2genes = np.dot(OT, rna)
    dataset_reconst = pd.DataFrame(spot2genes, columns=genes_names)
    return dataset_reconst


def pretrain_ae(model, optimizer, train_data, config, model_path):
    print('\n===========> pretraining... <===========')
    train_dataset = TrainDataset(train_data)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=config['batch_size'], drop_last=False)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    early_stopper = EarlyStopper(patience=args.patience, min_delta=0)
    history = []

    model.train()
    for epoch in range(config['pretrain_epochs']):
        current_loss = 0
        count = 0
        for batch_no, batch_x in enumerate(train_loader):
            batch_x = torch.squeeze(batch_x).to(device)
            batch_x_hat, latent = model(batch_x)
            loss = model.loss_ae(batch_x, batch_x_hat)
            current_loss += loss.item()
            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        history.append((current_loss))

        if early_stopper.early_stop(current_loss):
            break
        if epoch % 10 == 0:
            print('pretrain epoch %d: loss=%.6f' % (epoch, current_loss / count))

    torch.save(model.state_dict(), model_path)


def train(model, optimizer, x1, x2,type_mask,config, model_path):
    if not isinstance(x1, torch.Tensor):
        x1 = torch.from_numpy(x1).to(device)
    if not isinstance(x2, torch.Tensor):
        x2 = torch.from_numpy(x2).to(device)
    # training
    print('\n===========> training... <===========')
    early_stopper = EarlyStopper(patience=args.patience, min_delta=0)
    for epoch in range(config['epochs']):
        current_loss = 0
        x1_hat, x2_hat, z1, z2, z2_map = model(x1, x2)
        loss, loss_ae, loss_exp, reg, loss_ssim,loss_clr = model.loss_fn(x1, x2, x1_hat, x2_hat, z1, z2, z2_map,
                                                                                      type_mask,
                                                               weight_map=config['weight_map'],
                                                               weight_coef=config['weight_coef'],
                                                               weight_ssim=config['weight_ssim'],
                                                               weight_con=config['weight_con'] )

        current_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print('train epoch %d: loss=%.6f, loss_ae=%.6f, loss_map=%.6f, loss_coef=%.6f, loss_ssim=%.6f,loss_clr=%.6f' % (epoch, loss.item(), loss_ae.item(), \
                                                                                                         loss_exp.item(), reg.item(), loss_ssim.item(),loss_clr.item()))

    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()