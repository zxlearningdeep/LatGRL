# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.preprocess import sparse_mx_to_torch_sparse_tensor, normalize_adj_from_tensor, add_self_loop_and_normalize, remove_self_loop

EPS = 1e-15

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(GCN, self).__init__()
        self.weight_1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight_1 = self.reset_parameters(self.weight_1)
        if dropout:
            self.gc_drop = nn.Dropout(dropout)
        else:
            self.gc_drop = lambda x: x
        self.bc = nn.BatchNorm1d(output_dim)

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        return weight

    def forward(self, feat, adj, nlayer, sampler=False):
        if not sampler:
            z = self.gc_drop(torch.mm(feat, self.weight_1))
            z = torch.mm(adj, z)
            for i in range(1,nlayer):
                z = torch.mm(adj, z)

            outputs = F.normalize(z, dim=1)
            return outputs
        else:
            z_ = self.gc_drop(torch.mm(feat, self.weight_1))
            z = torch.mm(adj, z_)
            for i in range(1, nlayer):
                z = torch.mm(adj, z)

            return z

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop=0.1):
        super(Attention, self).__init__()

        self.act = nn.ELU()
        self.att_l = nn.Parameter(torch.empty(
            size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_l.data, gain=1.414)
        self.att_h = nn.Parameter(torch.empty(
            size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_h.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds_l, embeds_h):
        beta_l = []
        beta_h = []
        attn_curr_l = self.attn_drop(self.att_l)
        attn_curr_h = self.attn_drop(self.att_h)

        for embed in embeds_l:
            beta_l.append(attn_curr_l.matmul(embed.t()))
        for embed in embeds_h:
            beta_h.append(attn_curr_h.matmul(embed.t()))

        beta = beta_l + beta_h
        embeds = embeds_l + embeds_h
        beta = torch.cat(beta, dim=0)
        beta = self.act(beta)
        beta = self.softmax(beta)

        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i].reshape(-1,1)
        return z_mp
