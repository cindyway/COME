import scanpy as sc
import torch
from torch.utils.data import RandomSampler, SequentialSampler


def load_data(data):
    if  data=='dro':
        RNA_adata = sc.read_h5ad(r"data/dro_rna.h5ad")
        Spatial_adata = sc.read_h5ad(r"data/dro_st.h5ad")


    elif data=='smFISH':
        RNA_adata = sc.read_h5ad(r"data/smart_seq_adata.h5ad")
        Spatial_adata = sc.read_h5ad(r"data/smfish.h5ad")


    elif data == 'MERFISH':
        RNA_adata =sc.read_h5ad(r"data/smart_seq_adata.h5ad")
        Spatial_adata = sc.read_h5ad(r"data/merfish.h5ad")

    elif data == 'STARmap':
        RNA_adata =sc.read_h5ad(r"data/smart_seq_adata.h5ad")
        Spatial_adata = sc.read_h5ad(r"data/starmap.h5ad")

    elif data == 'STARmap':
        RNA_adata = sc.read_h5ad(r"data/smart_seq_adata.h5ad")
        Spatial_adata = sc.read_h5ad(r"data/starmap.h5ad")

    elif data == 'PDAC':
        RNA_adata = sc.read_h5ad(r"data/PDAC_sc.h5ad")
        Spatial_adata = sc.read_h5ad(r"data/PDAC_st.h5ad")


    else:
        raise Exception('Undefined data_name')

    sc.pp.filter_genes(RNA_adata, min_cells=1)
    sc.pp.filter_cells(RNA_adata, min_genes=1)

    return RNA_adata,Spatial_adata


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, train_data):
        self.data = train_data


    def __getitem__(self, index):
        # idx_1 = torch.randint(0, self.)

        return self.data[index]#,self.label[index]


    def __len__(self):
        # return the total size of data
        return self.data.shape[0]


class Data_Sampler(object):

    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)
        else:
            self.sampler = SequentialSampler(pairs)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = [batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size