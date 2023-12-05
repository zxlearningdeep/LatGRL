import numpy as np
import scipy.sparse as sp
import torch as th
import torch
# from sklearn.preprocessing import OneHotEncoder
from module.preprocess import *
import math
from torch.utils.data import RandomSampler


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new


def sp_tensor_to_sp_csr(adj):
    adj = adj.coalesce()
    row = adj.indices()[0]
    col = adj.indices()[1]
    data = adj.values()
    shape = adj.size()
    adj = sp.csr_matrix((data, (row, col)), shape=shape)
    return adj



def get_top_k(sim_l, sim_h, k1, k2):
    _, k_indices_pos = torch.topk(sim_l, k=k1, dim=1)
    _, k_indices_neg = torch.topk(sim_h, k=k2, dim=1)

    source = torch.tensor(range(len(sim_l))).reshape(-1, 1).to(sim_l.device)
    k_source_l = source.repeat(1, k1).flatten()
    k_source_h = source.repeat(1, k2).flatten()

    k_indices_pos = k_indices_pos.flatten()
    k_indices_pos = torch.stack((k_source_l, k_indices_pos), dim=0)

    k_indices_neg = k_indices_neg.flatten()
    k_indices_neg = torch.stack((k_source_h, k_indices_neg), dim=0)

    kg_pos = torch.sparse.FloatTensor(k_indices_pos, torch.ones((len(k_indices_pos[0]))).to(sim_l.device), sim_l.shape).coalesce()
    kg_neg = torch.sparse.FloatTensor(k_indices_neg, torch.ones((len(k_indices_neg[0]))).to(sim_l.device), sim_h.shape).coalesce()

    return kg_pos, kg_neg



p_num = 736389
a_num = 1134649
i_num = 8740
f_num = 59965


path = './data/mag/'
feat_p = th.load(path + 'paper_features.pt')

x_norm = torch.norm(feat_p, dim=-1, keepdim=True)
zero_indices = torch.nonzero(x_norm.flatten() == 0)
x_norm[zero_indices] += EPS

pap = th.load(path + 'pap.pt').coalesce()
pp = th.load(path + 'pp.pt').coalesce()
pp = (pp + pp.t()) / 2
pp = pp.coalesce()
adjs = [pap, pp]

# ones values:
adjs_new = []
for adj in adjs:
    adj = adj.coalesce()
    diag_index = torch.nonzero(adj.indices()[0] != adj.indices()[1]).flatten()
    adj = torch.sparse.FloatTensor(torch.stack((adj.indices()[0][diag_index], adj.indices()[1][diag_index]), dim=0),
                                   adj.values()[diag_index], adj.shape)
    adj = sparse_tensor_add_self_loop(adj)
    adjs_new.append(adj)

adjs = adjs_new

adjs = [normalize_adj_from_tensor(adj, mode='row', sparse=True).coalesce() for adj in adjs]
print('start rw')

rw_ = adjs[0].coalesce()
for i in range(1, len(adjs)):
    adj_rw = adjs[i].coalesce()
    rw_values = rw_.values() * 0.5
    rw_values = torch.cat((rw_values, adj_rw.values() * 0.5), dim=0)
    rw_indices = torch.cat((rw_.indices(), adj_rw.indices()), dim=1)
    rw = torch.sparse.FloatTensor(rw_indices, rw_values, rw_.shape).coalesce()




# latent graph construction
gc_batch_size = 1000
sampler_num = math.ceil(p_num / gc_batch_size)


graph_k = 5
anchor_num = 5000


sampler = RandomSampler(range(p_num), replacement=False)
sampler_ = [i for i in sampler]
print('start index')
anchor_index = th.tensor(sampler_[:anchor_num])
rw_anchor = torch.index_select(rw, dim=0, index=anchor_index)


# cuda
feat_p = feat_p.cuda()
x_norm = x_norm.cuda()
# adjs = [adj.cuda() for adj in adjs]
# rw = rw.cuda()
anchor_index = anchor_index.cuda()
rw_anchor = rw_anchor.cuda()

adjs_l_index = [[],[]]
adjs_h_index = [[],[]]
pos_index = [[],[]]

for i in range(sampler_num):
    print('epoch:', i)

    seed_node = torch.tensor(range(i*gc_batch_size, (i+1)*gc_batch_size))
    if (i+1)*gc_batch_size > p_num:
        seed_node = torch.tensor(range(i * gc_batch_size, p_num))

    pap_neighbor = torch.index_select(adjs[0], dim=0, index=seed_node)._indices()[1].unique()
    pp_neighbor = torch.index_select(adjs[1], dim=0, index=seed_node)._indices()[1].unique()
    neighbor_index = torch.cat((pap_neighbor, pp_neighbor), dim=0).unique()
    print(neighbor_index.shape)

    neighbor_rw = torch.index_select(rw, 0, neighbor_index).cuda()
    seed_node_rw = torch.index_select(rw, 0, seed_node).cuda()
    neighbor_index = neighbor_index.cuda()
    seed_node = seed_node.cuda()


    adj_sim_l = torch.sparse.mm(seed_node_rw, neighbor_rw.t()).to_dense()
    dot_numerator_l = torch.mm(feat_p[seed_node], feat_p[neighbor_index].t())
    dot_denominator_l = torch.mm(x_norm[seed_node], x_norm[neighbor_index].t())
    feat_sim_l = dot_numerator_l / dot_denominator_l
    sim_l = feat_sim_l * adj_sim_l


    adj_sim_h = torch.sparse.mm(seed_node_rw, rw_anchor.t()).to_dense()
    dot_numerator_h = torch.mm(feat_p[seed_node], feat_p[anchor_index].t())
    dot_denominator_h = torch.mm(x_norm[seed_node], x_norm[anchor_index].t())
    feat_sim_h = dot_numerator_h / dot_denominator_h
    sim_h = (1.0 - feat_sim_h) * (1.0 - adj_sim_h)

    adj_l, adj_h = get_top_k(sim_l, sim_h, graph_k+1, graph_k)

    adj_l_indices_0 = adj_l.indices()[0]
    adj_l_indices_1 = adj_l.indices()[1]
    adj_l_indices_0 = seed_node[adj_l_indices_0].cpu()
    adj_l_indices_1 = neighbor_index[adj_l_indices_1].cpu()
    adjs_l_index[0].append(adj_l_indices_0)
    adjs_l_index[1].append(adj_l_indices_1)

    adj_h_indices_0 = adj_h.indices()[0]
    adj_h_indices_1 = adj_h.indices()[1]
    adj_h_indices_0 = seed_node[adj_h_indices_0].cpu()
    adj_h_indices_1 = anchor_index[adj_h_indices_1].cpu()
    adjs_h_index[0].append(adj_h_indices_0)
    adjs_h_index[1].append(adj_h_indices_1)


adjs_l_index[0] = torch.cat(adjs_l_index[0],dim=0)
adjs_l_index[1] = torch.cat(adjs_l_index[1],dim=0)
adjs_h_index[0] = torch.cat(adjs_h_index[0],dim=0)
adjs_h_index[1] = torch.cat(adjs_h_index[1],dim=0)


lg_l = torch.sparse.FloatTensor(torch.stack(adjs_l_index, dim=0), torch.ones(len(adjs_l_index[0])), ([p_num, p_num])).coalesce()
lg_h = torch.sparse.FloatTensor(torch.stack(adjs_h_index, dim=0), torch.ones(len(adjs_h_index[0])), ([p_num, p_num])).coalesce()

diag_index = torch.nonzero(lg_l.indices()[0] != lg_l.indices()[1]).flatten()
lg_l = torch.sparse.FloatTensor(torch.stack((lg_l.indices()[0][diag_index], lg_l.indices()[1][diag_index]), dim=0),
                                lg_l.values()[diag_index], lg_l.shape)
lg_l = sparse_tensor_add_self_loop(lg_l).coalesce()

mp2vec_emb = torch.load(path + 'paper_mp2vec.pt').cuda()
feat = torch.cat((feat_p.cuda(), mp2vec_emb), dim=1)
adjs_l_filter, adjs_h_filter, lg_l_filter, lg_h_filter = pre_filter(feat, adjs, lg_l, lg_h, 1, 'cuda:0')

torch.save(lg_l_filter,  './data/mag/pre_filter/lg_l_filter.pt')
torch.save(lg_h_filter,  './data/mag/pre_filter/lg_h_filter.pt')
torch.save(adjs_l_filter,  './data/mag/pre_filter/adjs_l_filter.pt')
torch.save(adjs_h_filter,  './data/mag/pre_filter/adjs_h_filter.pt')
