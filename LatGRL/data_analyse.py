import numpy as np
import torch
import torch as th
import warnings
import matplotlib.pyplot as plt
import datetime
import pickle as pkl
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from utils.load_data import load_data
from utils.params import set_params
from module.preprocess import remove_self_loop
warnings.filterwarnings('ignore')

EPS = 1e-15


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def get_knn(x, k):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    zero_indices = torch.nonzero(x_norm.flatten() == 0)
    x_norm[zero_indices] += EPS
    dot_numerator = torch.mm(x, x.t())
    dot_denominator = torch.mm(x_norm, x_norm.t())
    sim_matrix = dot_numerator / dot_denominator

    sim_diag = torch.diag_embed(torch.diag(sim_matrix))
    sim_matrix = sim_matrix - sim_diag
    _, k_indices_pos = torch.topk(sim_matrix, k=k, dim=1)
    _, k_indices_neg = torch.topk(-sim_matrix, k=k, dim=1)

    source = torch.tensor(range(len(x))).reshape(-1,1).to(x.device)
    k_source = source.repeat(1,k).flatten()

    k_indices_pos = k_indices_pos.flatten()
    k_indices_pos = torch.stack((k_source,k_indices_pos),dim=0)

    k_indices_neg = k_indices_neg.flatten()
    k_indices_neg = torch.stack((k_source,k_indices_neg),dim=0)

    kg_pos = torch.sparse.FloatTensor(k_indices_pos, torch.ones((len(k_indices_pos[0]))).to(x.device), ([len(x),len(x)]))
    kg_eng = torch.sparse.FloatTensor(k_indices_neg, torch.ones((len(k_indices_neg[0]))).to(x.device), ([len(x),len(x)]))

    return kg_pos, kg_eng


def get_top_k(sim_l, sim_h, k):
    _, k_indices_pos = torch.topk(sim_l, k=k, dim=1)
    _, k_indices_neg = torch.topk(sim_h, k=k, dim=1)

    source = torch.tensor(range(len(x))).reshape(-1, 1).to(x.device)
    k_source = source.repeat(1, k).flatten()

    k_indices_pos = k_indices_pos.flatten()
    k_indices_pos = torch.stack((k_source, k_indices_pos), dim=0)

    k_indices_neg = k_indices_neg.flatten()
    k_indices_neg = torch.stack((k_source, k_indices_neg), dim=0)

    # kg_pos = torch.sparse.FloatTensor(k_indices_pos, torch.ones((len(k_indices_pos[0]))).to(x.device), ([len(x), len(x)]))
    # kg_neg = torch.sparse.FloatTensor(k_indices_neg, torch.ones((len(k_indices_neg[0]))).to(x.device), ([len(x), len(x)]))

    homo_r_l = len(torch.nonzero(label[k_indices_pos[0]] == label[k_indices_pos[1]])) / len(k_indices_pos[0])
    homo_r_h = len(torch.nonzero(label[k_indices_neg[0]] == label[k_indices_neg[1]])) / len(k_indices_neg[0])
    return homo_r_l, homo_r_h


def normalize_adj_from_tensor(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EPS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EPS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EPS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


args = set_params()
feat, adjs, label, idx_train, idx_val, idx_test = \
    load_data(args.dataset, args.ratio, args.type_num)
nb_classes = label.shape[-1]

x = feat
print('x', x.shape)

adjs = remove_self_loop(adjs)
for i in range(len(adjs)):
    adj = adjs[i]
    edge_num = adj._nnz()
    adj_indices = adj._indices()
    adjs[i] = torch.sparse.FloatTensor(adj_indices, torch.ones(edge_num), adj.shape).coalesce()
label = torch.topk(label, 1)[1].squeeze(1).cuda()
print(label.max())
x = x.cuda()
adjs = [adj.cuda() for adj in adjs]


adjs_indices = [adj.indices() for adj in adjs]
homo_r = [len(torch.nonzero(label[indices[0]]==label[indices[1]])) / len(indices[0]) for indices in adjs_indices]
print('homo:', homo_r)

x_norm = torch.norm(x, dim=-1, keepdim=True)
zero_indices = torch.nonzero(x_norm.flatten() == 0)
x_norm[zero_indices] += EPS

dot_numerator = torch.mm(x, x.t())
dot_denominator = torch.mm(x_norm, x_norm.t())
sim_matrix = dot_numerator / dot_denominator




n_rw = 0
# k=1
# adjs_rw_o = []
# for i in range(0, len(adjs)):
#     adj_ = adjs[i].to_dense()
#     adj_ += torch.eye(len(x)).cuda()
#     RW = adj_ / adj_.sum(dim=1)[:,None]
#
#     adjs_rw_o.append(RW)
# adjs_rw = torch.stack(adjs_rw_o, dim=0).mean(dim=0)
#
# adj_norm = torch.norm(adjs_rw, dim=1, keepdim=True)
# adj_sim = torch.mm(adjs_rw, adjs_rw.t()) / torch.mm(adj_norm, adj_norm.t())
#
# sim_fea_l = sim_matrix
# sim_fea_l = sim_fea_l - torch.diag_embed(torch.diag(sim_fea_l))
# sim_fea_h = 1-sim_matrix
#
# sim_adj_l = adj_sim
# sim_adj_l = sim_adj_l - torch.diag_embed(torch.diag(sim_adj_l))
# sim_adj_h = 1-adj_sim
#
#
#
# sim_l = adj_sim * sim_matrix
# sim_l = sim_l - torch.diag_embed(torch.diag(sim_l))
# sim_h = (1-adj_sim) * (1-sim_matrix)
#
#
# homo_fea = get_top_k(sim_fea_l, sim_fea_h, k)
# homo_adj = get_top_k(sim_adj_l, sim_adj_h, k)
# homo_ = get_top_k(sim_l, sim_h, k)



sim_fea_l = sim_matrix
sim_fea_l = sim_fea_l - torch.diag_embed(torch.diag(sim_fea_l))
sim_fea_h = 1-sim_matrix

k = 1
homo_fea = get_top_k(sim_fea_l, sim_fea_h, k)
print('fea knn k=',k,'  ',homo_fea)

k = 5
homo_fea = get_top_k(sim_fea_l, sim_fea_h, k)
print('fea knn k=',k,'  ',homo_fea)


k = 10
homo_fea = get_top_k(sim_fea_l, sim_fea_h, k)
print('fea knn k=',k,'  ',homo_fea)


# dblp
# homo: [0.7987528344671202, 0.6696854839387579, 0.32449768549362923]
# fea knn k= 1    (0.7453783583929011, 0.23440966231205324)
# fea knn k= 5    (0.6838550653192014, 0.21326103031796895)
# fea knn k= 10    (0.6591323638156273, 0.20909539068277053)

# acm
# homo: [0.8085224950774603, 0.6392921036760237]
# fea knn k= 1    (0.78626524010948, 0.1565065936800199)
# fea knn k= 5    (0.7417267977108734, 0.26295098283155016)
# fea knn k= 10    (0.7201045036078626, 0.27653645185369496)

# imdb
# homo: [0.6140643985419199, 0.44427725703009374]
# fea knn k= 1    (0.5053763440860215, 0.2653108929406265)
# fea knn k= 5    (0.4690977092099112, 0.26367461430575034)
# fea knn k= 10    (0.45425432445067787, 0.2754324450677887)

# yelp
# homo: [0.6407566861728862, 0.44969356194766014, 0.3876386482853009]
# fea knn k= 1    (0.8894414690130069, 0.11553175210405509)
# fea knn k= 5    (0.8869166029074216, 0.16143840856924255)
# fea knn k= 10    (0.8825172149961744, 0.18557765876052026)

