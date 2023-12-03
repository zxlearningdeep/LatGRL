import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.preprocess import remove_self_loop, normalize_adj_from_tensor

EPS = 1e-15


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

    kg_pos = torch.sparse.FloatTensor(k_indices_pos, torch.ones((len(k_indices_pos[0]))).to(sim_l.device), ([len(sim_l), len(sim_l)]))
    kg_neg = torch.sparse.FloatTensor(k_indices_neg, torch.ones((len(k_indices_neg[0]))).to(sim_l.device), ([len(sim_l), len(sim_l)]))

    return kg_pos, kg_neg


def graph_construction(x, adjs, k1, k2, k_pos):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    zero_indices = torch.nonzero(x_norm.flatten() == 0)
    x_norm[zero_indices] += EPS
    dot_numerator = torch.mm(x, x.t())
    dot_denominator = torch.mm(x_norm, x_norm.t())
    fea_sim = dot_numerator / dot_denominator

    adjs_rw = []
    for i in range(0, len(adjs)):
        adj_ = adjs[i].to_dense()
        adj_ += torch.eye(len(x)).to(x.device)
        RW = adj_ / adj_.sum(dim=1)[:,None]

        adjs_rw.append(RW)

    adjs_rw = torch.stack(adjs_rw, dim=0).mean(dim=0)
    adj_norm = torch.norm(adjs_rw, dim=1, keepdim=True)
    zero_indices = torch.nonzero(adj_norm.flatten() == 0)
    adj_norm[zero_indices] += EPS
    adj_sim = torch.mm(adjs_rw, adjs_rw.t()) / torch.mm(adj_norm, adj_norm.t())

    sim_l = adj_sim * fea_sim
    sim_l = sim_l - torch.diag_embed(torch.diag(sim_l))
    sim_h = (1 - adj_sim) * (1 - fea_sim)

    kg_pos, kg_neg = get_top_k(sim_l, sim_h, k1, k2)
    if k_pos <= 0:
        pos = torch.eye(len(x)).to_sparse().to(x.device)
    else:
        pos, _ = get_top_k(sim_l, sim_h, k_pos, k_pos)
        pos = (pos.to_dense() + torch.eye(len(x)).to(x.device)).to_sparse()
    return [kg_pos], [kg_neg], [pos]



def graph_process(adjs, feat, args):
    adjs = [adj.coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)  # return sparse tensor
    adj_I = torch.eye(len(feat)).to(feat.device)

    adjs_l, adjs_h, pos = graph_construction(feat, adjs, args.graph_k, args.graph_k, args.k_pos)

    adjs_l = [normalize_adj_from_tensor(adj_I+adj.to_dense(), mode='sym') for adj in adjs_l]
    adjs_h = [normalize_adj_from_tensor(adj.to_dense(), mode='sym').to_sparse() for adj in adjs_h]
    adjs_h = [adj_I-adj.to_dense() for adj in adjs_h]
    adjs_o = [normalize_adj_from_tensor(adj_I+adj.to_dense(), mode='sym').to_sparse() for adj in adjs]
    print(adjs_l)
    print(adjs_h)
    print(adjs_o)

    return adjs_l, adjs_h, adjs_o, pos


