# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import random
from torch import Tensor

EPS = 1e-15


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm_coo = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().todense()

    adj_torch = torch.from_numpy(adj_norm_coo).float()
    if torch.cuda.is_available():
        adj_torch = adj_torch.cuda()
    return adj_torch

def Count(feats,num):
    count = 0
    for i in range(feats.size()[0]):
        for j in range(feats.size()[1]):
            if feats[i][j] == num:
                count = count + 1
    return count

def pathsim(adjs, max_nei):
    print("the number of edges:", [adj.getnnz() for adj in adjs])
    top_adjs = []
    adjs_num = []
    for t in range(len(adjs)):
        A = adjs[t].todense()
        value = []
        x,y = A.nonzero()
        for i,j in zip(x,y):
            value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
        pathsim_matrix = sp.coo_matrix((value, (x, y)), shape=A.shape).toarray()
        idx_x = np.array([np.ones(max_nei[t])*i for i in range(A.shape[0])], dtype=np.int32).flatten()
        idx_y = np.sort(np.argsort(pathsim_matrix, axis=1)[:,::-1][:,0:max_nei[t]]).flatten()
        new = []
        for i,j in zip(idx_x,idx_y):
            new.append(A[i,j])
        new = (np.int32(np.array(new)))
        adj_num = np.array(new).nonzero()
        adj_new = sp.coo_matrix((new[adj_num], (idx_x[adj_num],idx_y[adj_num])), shape=adjs[t].shape)
        adjs_num.append(adj_num[0].shape[0])
        top_adjs.append(adj_new)
    print("the top-k number of edges:", [adj for adj in adjs_num])
    return top_adjs

def spcoo_to_torchcoo(adj):
    values = adj.data
    indices = np.vstack((adj.row ,adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    adj_torchcoo = torch.sparse_coo_tensor(i, v, adj.shape)
    return adj_torchcoo


def mask_edges(adjs, sub_num, adj_mask):
    #print("the number of edges:", [adj.values for adj in adjs])
    mask_adjs = []
    for i in range(sub_num):    
        adj = adjs[i]
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        adj_tuple = sparse_to_tuple(adj)   
        edges = adj_tuple[0]  
        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(edges)
        mask_edges_num = int(adj_mask*len(edges))
        rest_edges = edges[mask_edges_num:]
        data = np.ones(rest_edges.shape[0])
        adj = sp.coo_matrix((data, (rest_edges[:, 0], rest_edges[:, 1])), shape=adjs[i].shape)
        adj = (adj + np.eye(adjs[i].shape[0]))
        adj = normalize_adj(adj)
        mask_adjs.append(adj)

      #  print("the number of mask edges:", [mask_adj.getnnz() for mask_adj in mask_adjs])

    return mask_adjs

def mask_feature(feat, adj_mask):
    feats_coo = sp.coo_matrix(feat)
    feats_num = feats_coo.getnnz()
    feats_idx = [i for i in range(feats_num)]
    mask_num = int(feats_num * adj_mask)
    mask_idx = random.sample(feats_idx, mask_num)
    feats_data = feats_coo.data
    for j in mask_idx:
        feats_data[j] = 0
    mask_feat = torch.sparse.FloatTensor(torch.LongTensor([feats_coo.row.tolist(), feats_coo.col.tolist()]), torch.FloatTensor(feats_data.astype(np.float64)))

    if torch.cuda.is_available():
        mask_feat = mask_feat.cuda()
    return mask_feat

def mask_features(feats, adj_mask):
    if len(feats.size()) == 3:
        mask_feats = []
        for feat in feats:
            mask_feats.append(mask_feature(feat,adj_mask))
        if torch.cuda.is_available():
            mask_feats =  [f.cuda() for f in mask_feats]
    else:
        mask_feats = mask_feature(feats, adj_mask)
        if torch.cuda.is_available():
            mask_feats = mask_feats.cuda() 
    return mask_feats

def txt_to_coo(dataset, relation):
    txt = np.genfromtxt("../data/" + dataset + "/" + relation + ".txt")
    row = txt[:,0]
    col = txt[:,1]
    data = np.ones(len(row))
    return torch.sparse.FloatTensor(torch.LongTensor([row.tolist(), col.tolist()]),
                            torch.FloatTensor(data.astype(np.float))).to_dense()



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


def remove_self_loop(adjs):
    adjs_ = []
    for i in range(len(adjs)):
        adj = adjs[i].coalesce()
        diag_index = torch.nonzero(adj.indices()[0] != adj.indices()[1]).flatten()
        adj = torch.sparse.FloatTensor(adj.indices()[:, diag_index], adj.values()[diag_index], adj.shape).coalesce()
        adjs_.append(adj)
    return adjs_


def add_self_loop_and_normalize(indices, values, nnodes):
    i_indices = torch.stack((torch.tensor(range(0, nnodes)), torch.tensor(range(0, nnodes))), dim=0).to(indices.device)
    i_values = torch.ones(nnodes).to(indices.device)
    edge_indices = torch.cat((i_indices, indices), dim=1)
    edge_values = torch.cat((i_values, values),dim=0)
    edge = torch.sparse.FloatTensor(edge_indices, edge_values, ([nnodes,nnodes]))
    edge = normalize_adj_from_tensor(edge, mode='sym', sparse=True)
    return edge


def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new


def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return torch.sparse.FloatTensor(index, torch.ones(len(index[0])), adj.shape)


def pre_filter(feat, adjs, lg_l, lg_h, nlayer, device):
    adjs = [normalize_adj_from_tensor(adj, mode='row', sparse=True).coalesce().to(device) for adj in adjs]
    lg_l = normalize_adj_from_tensor(lg_l, mode='row', sparse=True).coalesce().to(device)
    lg_h = normalize_adj_from_tensor(lg_h, mode='row', sparse=True).coalesce().to(device)
    feat = feat.to(device)

    num_sub_graph = len(adjs)
    node_num = len(feat)
    index_I = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0)
    values_I = torch.ones(node_num)
    adj_I = torch.sparse.FloatTensor(index_I, values_I, ([node_num, node_num])).coalesce().to(device)

    adjs_l = adjs
    adjs_h = [(adj_I - adj).coalesce() for adj in adjs]
    lg_h = (adj_I - lg_h).coalesce()
    print(adjs_h)
    print(lg_h)


    adjs_l_filter, adjs_h_filter, lg_l_filter, lg_h_filter = [[] for _ in range(num_sub_graph)], [[] for _ in range(num_sub_graph)], [], []

    h_l, h_h, h_lg_l, h_lg_h = [feat for _ in range(num_sub_graph)], [feat for _ in range(num_sub_graph)], feat, feat
    for i in range(nlayer):
        print('pre filter layer: ', i)
        h_l_ = [torch.sparse.mm(adjs_l[j], h_l[j]) for j in range(num_sub_graph)]
        h_h_ = [torch.sparse.mm(adjs_h[j], h_h[j]) for j in range(num_sub_graph)]
        h_l = h_l_
        h_h = h_h_

        h_lg_l = torch.sparse.mm(lg_l, h_lg_l)
        h_lg_h = torch.sparse.mm(lg_h, h_lg_h)

        lg_l_filter.append(h_lg_l.cpu())
        lg_h_filter.append(h_lg_h.cpu())

        for j in range(num_sub_graph):
            adjs_l_filter[j].append(h_l[j].cpu())
            adjs_h_filter[j].append(h_h[j].cpu())


    return adjs_l_filter, adjs_h_filter, lg_l_filter, lg_h_filter


def find_idx(a: Tensor, b: Tensor, missing_values: int = -1):
    """Find the first index of b in a, return tensor like a."""
    a, b = a.clone(), b.clone()
    invalid = ~torch.isin(a, b)
    a[invalid] = b[0]
    sorter = torch.argsort(b)
    b_to_a: Tensor = sorter[torch.searchsorted(b, a, sorter=sorter)]
    b_to_a[invalid] = missing_values
    return b_to_a



def pos_samplering(seed, pos, device):
    pos_ = torch.index_select(pos, dim=0, index=seed).coalesce().indices()
    all_index = torch.cat((pos_[0], pos_[1]), dim=0).unique()
    pos_0 = pos_[0].tolist()
    pos_1 = pos_[1].tolist()

    key = all_index.tolist()
    values = [i for i in range(len(all_index))]
    index_dict = dict(zip(key, values))

    pos_0_in_all = torch.tensor([index_dict[i] for i in pos_0])
    pos_1_in_all = torch.tensor([index_dict[i] for i in pos_1])
    pos_graph = torch.sparse.FloatTensor(torch.stack((pos_0_in_all, pos_1_in_all), dim=0), torch.ones(len(pos_0_in_all)), ([len(all_index), len(all_index)])).coalesce()

    seed_in_all = pos_0_in_all.unique()
    return pos_graph.to(device), all_index, seed_in_all.to(device)
