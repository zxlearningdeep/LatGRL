import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from module.preprocess import remove_self_loop, normalize_adj_from_tensor, sparse_tensor_add_self_loop
from torch.utils.data import RandomSampler

EPS = 1e-15


def find_idx(a: Tensor, b: Tensor, missing_values: int = -1):
    """Find the first index of b in a, return tensor like a."""
    a, b = a.clone(), b.clone()
    invalid = ~torch.isin(a, b)
    a[invalid] = b[0]
    sorter = torch.argsort(b)
    b_to_a: Tensor = sorter[torch.searchsorted(b, a, sorter=sorter)]
    b_to_a[invalid] = missing_values
    return b_to_a



class graph_sampler_mag():
    def __init__(self, feat_seed, feat_adjs, feat_lg, adjs, lg, args, cuda=True):
        super(graph_sampler_mag, self).__init__()

        self.feat_seed = feat_seed
        self.feat_pap_l = feat_adjs[0][0]
        self.feat_pp_l = feat_adjs[0][1]
        self.feat_pap_h = feat_adjs[1][0]
        self.feat_pp_h = feat_adjs[1][1]
        self.feat_lg_l = feat_lg[0]
        self.feat_lg_h = feat_lg[1]

        self.pap_l = adjs[0][0]
        self.pp_l = adjs[0][1]
        self.pap_h = adjs[1][0]
        self.pp_h = adjs[1][1]
        self.lg_l = lg[0]
        self.lg_h = lg[1]

        self.cuda = cuda



    def get_sampler(self, i):

        feat_seed = torch.tensor(self.feat_seed[i]).float()

        feat_pap_l = torch.tensor(self.feat_pap_l[i]).float()
        feat_pp_l = torch.tensor(self.feat_pp_l[i]).float()
        feat_pap_h = torch.tensor(self.feat_pap_h[i]).float()
        feat_pp_h = torch.tensor(self.feat_pp_h[i]).float()
        feat_lg_l = torch.tensor(self.feat_lg_l[i]).float()
        feat_lg_h = torch.tensor(self.feat_lg_h[i]).float()

        pap_l = torch.tensor(self.pap_l[i]).t()
        pap_l = torch.sparse.FloatTensor(pap_l, torch.ones(len(pap_l[0])), ([len(feat_seed), len(feat_pap_l)]))

        pp_l = torch.tensor(self.pp_l[i]).t()
        pp_l = torch.sparse.FloatTensor(pp_l, torch.ones(len(pp_l[0])), ([len(feat_seed), len(feat_pp_l)]))

        pap_h = torch.tensor(self.pap_h[i]).t()
        pap_h = torch.sparse.FloatTensor(pap_h, torch.ones(len(pap_h[0])), ([len(feat_seed), len(feat_pap_h)]))

        pp_h = torch.tensor(self.pp_h[i]).t()
        pp_h = torch.sparse.FloatTensor(pp_h, torch.ones(len(pp_h[0])), ([len(feat_seed), len(feat_pp_h)]))

        lg_l = torch.tensor(self.lg_l[i]).t()
        lg_l = torch.sparse.FloatTensor(lg_l, torch.ones(len(lg_l[0])), ([len(feat_seed), len(feat_lg_l)]))

        lg_h = torch.tensor(self.lg_h[i]).t()
        lg_h = torch.sparse.FloatTensor(lg_h, torch.ones(len(lg_h[0])), ([len(feat_seed), len(feat_lg_h)]))

        pap_l_ = normalize_adj_from_tensor(pap_l, mode='row', sparse=True).coalesce()
        pp_l_ = normalize_adj_from_tensor(pp_l, mode='row', sparse=True).coalesce()
        pap_h_ = normalize_adj_from_tensor(pap_h, mode='row', sparse=True).coalesce()
        pp_h_ = normalize_adj_from_tensor(pp_h, mode='row', sparse=True).coalesce()
        lg_l_ = normalize_adj_from_tensor(lg_l, mode='row', sparse=True).coalesce()
        lg_h_ = normalize_adj_from_tensor(lg_h, mode='row', sparse=True).coalesce()

        if self.cuda:
            feat_seed = feat_seed.cuda()

            feat_pap_l = feat_pap_l.cuda()
            feat_pp_l = feat_pp_l.cuda()
            feat_pap_h = feat_pap_h.cuda()
            feat_pp_h = feat_pp_h.cuda()
            feat_lg_l = feat_lg_l.cuda()
            feat_lg_h = feat_lg_h.cuda()

            pap_l_ = pap_l_.cuda()
            pp_l_ = pp_l_.cuda()
            pap_h_ = pap_h_.cuda()
            pp_h_ = pp_h_.cuda()
            lg_l_ = lg_l_.cuda()
            lg_h_ = lg_h_.cuda()

        return feat_seed,  [[feat_pap_l, feat_pp_l], [feat_pap_h, feat_pp_h]], [feat_lg_l, feat_lg_h], [[pap_l_, pp_l_], [pap_h_, pp_h_]], [lg_l_, lg_h_]


class Node_Sampler():
    def __init__(self, adjs_l_filter, adjs_h_filter, lg_l_filter, lg_h_filter, feat, num_node, batchsize, cuda=True):
        super(Node_Sampler, self).__init__()

        self.adjs_l_filter = adjs_l_filter
        self.adjs_h_filter = adjs_h_filter
        self.lg_l_filter = lg_l_filter
        self.lg_h_filter = lg_h_filter
        self.feat = feat

        self.sampler_test = torch.tensor(range(num_node))

        self.num_target_node = num_node
        self.batchsize = batchsize
        self.cuda = cuda

    def random_sampler(self):

        sampler_ = RandomSampler(range(self.num_target_node), replacement=False)
        sampler = [i for i in sampler_]
        self.sampler = torch.tensor(sampler)

    def filter_sampler(self, batch_index):
        start_index = batch_index * self.batchsize
        end_index = start_index + self.batchsize
        if end_index > self.num_target_node:
            end_index = self.num_target_node
            start_index = end_index - self.batchsize
        seed_node = self.sampler[start_index:end_index]

        adjs_l_filter_sampler = [F.normalize(filters[-1][seed_node], p=2, dim=-1) for filters in self.adjs_l_filter]
        adjs_h_filter_sampler = [F.normalize(filters[-1][seed_node], p=2, dim=-1) for filters in self.adjs_h_filter]
        lg_l_filter_sampler = F.normalize(self.lg_l_filter[-1][seed_node], p=2, dim=-1)
        lg_h_filter_sampler = F.normalize(self.lg_h_filter[-1][seed_node], p=2, dim=-1)


        feat = self.feat[seed_node]

        if self.cuda:
            adjs_l_filter_sampler = [h.cuda() for h in adjs_l_filter_sampler]
            adjs_h_filter_sampler = [h.cuda() for h in adjs_h_filter_sampler]
            lg_l_filter_sampler = lg_l_filter_sampler.cuda()
            lg_h_filter_sampler = lg_h_filter_sampler.cuda()
            feat = feat.cuda()

        return adjs_l_filter_sampler, adjs_h_filter_sampler, lg_l_filter_sampler, lg_h_filter_sampler, feat

    def filter_sampler_test(self, batch_index):
        start_index = batch_index * self.batchsize
        end_index = start_index + self.batchsize
        if end_index > self.num_target_node:
            end_index = self.num_target_node
        seed_node = self.sampler_test[start_index:end_index]

        adjs_l_filter_sampler = [F.normalize(filters[-1][seed_node], p=2, dim=-1) for filters in self.adjs_l_filter]
        adjs_h_filter_sampler = [F.normalize(filters[-1][seed_node], p=2, dim=-1) for filters in self.adjs_h_filter]


        if self.cuda:
            adjs_l_filter_sampler = [h.cuda() for h in adjs_l_filter_sampler]
            adjs_h_filter_sampler = [h.cuda() for h in adjs_h_filter_sampler]

        return adjs_l_filter_sampler, adjs_h_filter_sampler


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


def graph_construction(x, adjs, k1, k2, k_pos, anchor_num):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    zero_indices = torch.nonzero(x_norm.flatten() == 0)
    x_norm[zero_indices] += EPS
    dot_numerator = torch.mm(x, x.t())
    dot_denominator = torch.mm(x_norm, x_norm.t())
    fea_sim = dot_numerator / dot_denominator

    adjs_rw = []
    adj_sum = 0
    for i in range(0, len(adjs)):
        adj_ = adjs[i].to_dense()
        adj_ += torch.eye(len(x)).to(x.device)
        RW = adj_ / adj_.sum(dim=1)[:,None]

        adjs_rw.append(RW)
        adj_sum += adj_

    adjs_rw = torch.stack(adjs_rw, dim=0).mean(dim=0)
    adj_norm = torch.norm(adjs_rw, dim=1, keepdim=True)
    zero_indices = torch.nonzero(adj_norm.flatten() == 0)
    adj_norm[zero_indices] += EPS
    adj_sim = torch.mm(adjs_rw, adjs_rw.t()) / torch.mm(adj_norm, adj_norm.t())

    sim_l = adj_sim * fea_sim
    sim_l = sim_l - torch.diag_embed(torch.diag(sim_l))
    sim_h = (1 - adj_sim) * (1 - fea_sim)

    # mask:
    adj_sum = adj_sum.to_sparse().coalesce()
    adj_sum = torch.sparse.FloatTensor(adj_sum.indices(), torch.ones(len(adj_sum.indices()[0])), adj_sum.shape).to_dense()
    sim_l = sim_l * adj_sum

    sampler = RandomSampler(range(len(x)), replacement=False)
    sampler_ = [i for i in sampler]
    anchor_index = torch.tensor(sampler_[:anchor_num])
    anchor_mask = torch.zeros((len(x), len(x)))
    anchor_mask[:,anchor_index] = 1.
    sim_h = sim_h * anchor_mask


    kg_pos, kg_neg = get_top_k(sim_l, sim_h, k1, k2)
    if k_pos <= 0:
        pos = torch.eye(len(x)).to_sparse().to(x.device)
    else:
        pos, _ = get_top_k(sim_l, sim_h, k_pos, k_pos)
        pos = (pos.to_dense() + torch.eye(len(x)).to(x.device)).to_sparse().coalesce()
    return [kg_pos], [kg_neg], [pos]



def graph_process_sampler(adjs, feat, args):
    adjs = [adj.coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)  # return sparse tensor
    adj_I = torch.eye(len(feat)).to(feat.device)

    if args.LG_construction:
        print('construct LG')
        adjs_l, adjs_h, pos = graph_construction(feat, adjs, args.graph_k, args.graph_k, args.k_pos, args.anchor_num)
        adjs_l = [(adj_I + adj.to_dense()).to_sparse() for adj in adjs_l]

        torch.save(adjs_l[0], './data/' + args.dataset + '/sampler/LG_L.pt')
        torch.save(adjs_h[0], './data/' + args.dataset + '/sampler/LG_H.pt')
    else:
        adjs_l = [torch.load('./data/'+args.dataset+'/sampler/LG_L.pt').coalesce()]
        adjs_h = [torch.load('./data/'+args.dataset+'/sampler/LG_H.pt').coalesce()]


    adjs_o = [(adj_I + adj.to_dense()).to_sparse() for adj in adjs]

    print(adjs_l)
    print(adjs_h)
    print(adjs_o)


    return adjs_l, adjs_h, adjs_o

