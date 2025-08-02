import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from CNN import CNN
import numpy as np


# ZPI*Spatial GCN Layer   and   ZPI*Temporal GCN Layers
class TLSGCN(nn.Module):
    def __init__(self, dim_in, dim_out, link_len, emb_dim, window_len):
        super(TLSGCN, self).__init__()
        self.link_len = link_len
        self.weights_pool = nn.Parameter(torch.randn(emb_dim, link_len, dim_in, int(dim_out / 2)))

        self.weights_window = nn.Parameter(torch.randn(emb_dim, dim_in, int(dim_out / 2)))  # int(dim_in/2)

        self.bias_pool = nn.Parameter(torch.randn(emb_dim, dim_out))
        self.T = nn.Parameter(torch.randn(window_len))  # window_len
        self.cnn = CNN(int(dim_out / 2))

    def forward(self, x, x_window, node_embeddings, zigzag_PI):
        node_num = node_embeddings.shape[1]
        node_embeddings = node_embeddings[0]

        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)

        support_set = [torch.eye(node_num).to(supports.device), supports]

        # laplacian link
        for k in range(2, self.link_len):
            support_set.append(torch.mm(supports, support_set[k - 1]))
        supports = torch.stack(support_set, dim=0)



        # spatial graph convolution
        # print(node_embeddings.shape)
        # print(self.weights_pool.shape)
        # print(torch.max(node_embeddings))
        # assert torch.isnan(node_embeddings).sum() == 0
        # assert torch.isnan(self.weights_pool).sum() == 0

        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, link_len, dim_in, dim_out/2
        
        # assert torch.isnan(weights).sum() ==0
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        

        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, link_len, N, dim_in
        #print(x_g.shape)
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, link_len, dim_in
        # print(torch.max(x_g))
        #print(x_g.shape)

        # print(torch.max(weights))
        #print(weights.shape)

        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights)  # B, N, dim_out/2(4,62,2,62),(62,2,10,31)
        # print(x_gconv)
        # assert torch.isnan(x_gconv).sum() == 0

        # temporal feature transformation
        weights_window = torch.einsum('nd,dio->nio', node_embeddings, self.weights_window)  # N, dim_in, dim_out/2
        # print(x_window.shape)
        # print(weights_window.shape)
        x_w = torch.einsum('btni,nio->btno', x_window, weights_window)  # B, T, N, dim_out/2
        # assert torch.isnan(x_w).sum() == 0
        # print(x_w.shape)
        x_w = x_w.permute(0, 2, 3, 1)  # B, N, dim_out/2, T
        # print(x_w)
        # print(self.T.shape)
        x_wconv = torch.matmul(x_w, self.T)  # B, N, dim_out/2, T
        x_wconv = torch.matmul(x_w, self.T)  # B, N, dim_out/2

        # print(x_wconv)

        # zigzag persistence representation learning

        topo_cnn = self.cnn(zigzag_PI) # B, dim_out/2, dim_out/2
        x_tgconv = torch.einsum('bno,bo->bno', x_gconv, topo_cnn)
        x_twconv = torch.einsum('bno,bo->bno', x_wconv, topo_cnn)

        # combination operation
        x_gwconv = torch.cat([x_tgconv, x_twconv], dim=-1) + bias  # B, N, dim_out
        return x_gwconv
