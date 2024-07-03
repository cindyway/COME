import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cal_ssim



class AutoEncoder(nn.Module):
    def __init__(self, dims):
        super(AutoEncoder, self).__init__()
        len_layer = len(dims) - 1
        self._encoder = nn.Sequential()


        for i in range(len_layer):
            self._encoder.add_module('Linear%d' % i, nn.Linear(dims[i], dims[i+1]))
            self._encoder.add_module('Relu%d' % i, nn.ReLU())


        self._decoder = nn.Sequential()
        dims = list(reversed(dims))
        for i in range((len_layer - 1)):
            self._decoder.add_module('Linear%d' % i, nn.Linear(dims[i], dims[i+1]))
            self._decoder.add_module('Relu%d' % i, nn.ReLU())

        self._decoder.add_module('Linear%d' % (len_layer - 1), nn.Linear(dims[len_layer - 1], dims[-1]))
        self._decoder.add_module('Sigmod%d' % (len_layer - 1), nn.Sigmoid())

    def encoder(self, x):
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

    def loss_ae(self, x, x_hat):
        return F.mse_loss(x_hat, x, reduction='mean')

class MapNet(nn.Module):
    def __init__(self, n, m):
        super(MapNet, self).__init__()
        self.Coefficient = nn.Parameter(torch.ones((n, m), dtype=torch.float32) / (n * m), requires_grad=True)
    def init_param(self,similarity):
        self.Coefficient.data=similarity
    def forward(self, x):
        y = torch.matmul(self.Coefficient, x)
        return y

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # params
        self._dims = config['dims']
        self._n1 = config['num_sample1']
        self._n2 = config['num_sample2']
        # net
        self.ae = AutoEncoder(self._dims)
        self.map = MapNet(self._n1, self._n2)
        self.tau=config['tau']
        self.clr_criterion = ContrastiveLoss(self.tau)



    def forward(self, x1, x2):

        x2_hat, z2 = self.ae(x2)
        z1 = self.ae.encoder(x1)
        z2_map = self.map(z2)
        x1_hat = self.ae.decoder(z2_map)
        return x1_hat, x2_hat, z1, z2, z2_map

    def st_mask(self, adj):
        '''neighbor contrastive mask'''
        # adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
        # adj[adj > 0] = 1

        return adj

    def cross_mask(self, Coefficient, num_st,num_rna):

        # if num_rows > num_cols:
        #     max_values, max_indices = Coefficient.max(dim=1)
        #     mask = torch.zeros_like(Coefficient)
        #     mask[torch.arange(num_rows), max_indices] = 1
        # else:
        #     max_values, max_indices = Coefficient.max(dim=0)
        #     mask = torch.zeros_like(Coefficient)
        #     mask[max_indices, torch.arange(num_cols)] = 1
        max_values, max_indices = Coefficient.max(dim=0)
        mask = torch.zeros_like(Coefficient)
        mask[max_indices, torch.arange(num_rna)] = 1

        return mask
    def cell_type_mask(self, sparse_matrix):

        cell_type_matrix= torch.from_numpy(sparse_matrix)
        #cell_type_matrix = cell_type_matrix - torch.diag_embed(cell_type_matrix.diag())  # remove self-loop

        return cell_type_matrix


    def loss_fn(self, x1, x2, x1_hat, x2_hat, z1, z2, z2_map,  type_matrix,weight_map, weight_coef, weight_ssim, weight_con,norm='l2'):

        G_pred = torch.matmul(self.map.Coefficient, x2)
        ssim_test_loss = 1 - cal_ssim(G_pred, x1, M=None)
        ssim_test_loss_all = ssim_test_loss
        loss_coef = torch.sum(torch.pow(self.map.Coefficient, 2))
        loss_ae = self.ae.loss_ae(x1, x1_hat) + self.ae.loss_ae(x2, x2_hat)
        loss_map = F.mse_loss(z2_map, z1, reduction='mean')

        cross_mask=self.cross_mask(self.map.Coefficient,self._n1,self._n2)
        type_mask=self.cell_type_mask(type_matrix)
        full_mask = torch.zeros(self._n1+self._n2, self._n1+self._n2)

        full_mask[:self._n1, :self._n1:]=torch.eye(self._n1)
        full_mask[:self._n1, -self._n2:] = cross_mask
        full_mask[-self._n2:, -self._n2:] =type_mask
        full_mask[-self._n2:, :self._n1] =  cross_mask.T

        loss_clr=self.clr_criterion(z1,z2,mask=full_mask)


        loss = loss_ae + weight_map * loss_map + loss_coef * weight_coef + ssim_test_loss_all*weight_ssim+weight_con*loss_clr
        return loss, loss_ae, weight_map * loss_map, loss_coef * weight_coef,ssim_test_loss_all*weight_ssim,weight_con*loss_clr

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self._temperature = temperature
        self.mask = None

    def mask_correlated_samples(self, batch_size, device='cuda'):
        N = 2 * batch_size
        if self.mask is None:
            mask = torch.eye(N, device=device)
            mask[torch.arange(0, N), (torch.arange(0, N) + batch_size) % N] = 1
            self.mask = mask
        return self.mask

    def compute_cosine_sim(self, h1, h2):
        sim = torch.tensordot(h1.unsqueeze(1), h2.T.unsqueeze(0), dims=2)
        # sim=torch.mm(h1, h2.t())
        return sim

    def mask_pos_and_neg(self, mask):
        negative_mask = 1 - mask
        negative_mask = negative_mask.fill_diagonal_(0)
        positive_mask = mask.fill_diagonal_(0)

        return positive_mask, negative_mask

    def forward(self, h1, h2, mask=None):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        h_cat = torch.cat([h1, h2], dim=0)
        sim = self.compute_cosine_sim(h_cat, h_cat) / self._temperature
        sim_exp = torch.exp(sim)
        if mask is None:
            mask = self.mask_correlated_samples(h1.shape[0], device=h1.device)
        positive_mask, negative_mask = self.mask_pos_and_neg(mask)

        # numerator=torch.diag(torch.mm(sim_exp, positive_mask.t()))+1e-12
        # denominator=torch.diag(torch.mm(sim_exp, (positive_mask + negative_mask).t()))+1e-12
        numerator=torch.sum(torch.mul(sim_exp, positive_mask),dim=1)+1e-12
        denominator=torch.sum(torch.mul(sim_exp, (positive_mask + negative_mask)),dim=1)+1e-12

        loss = torch.sum(-torch.log( numerator/ denominator )) / (h_cat.shape[0])
        return loss


