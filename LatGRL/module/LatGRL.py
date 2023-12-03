# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .preprocess import *
from .encoder import *
from .loss_fun import *


class LatGRL(nn.Module):
    def __init__(self, feats_dim, sub_num, hidden_dim, embed_dim, tau, dropout, act, drop_feat, nnodes, dataset, alpha, nlayer):
        super(LatGRL, self).__init__()
        self.alpha = alpha
        self.feats_dim = feats_dim
        self.embed_dim = embed_dim
        self.sub_num = sub_num
        self.tau = tau
        self.dataset = dataset
        self.nnodes = nnodes
        self.act = act
        self.nlayer = nlayer
        self.nlayer_c = nlayer
        self.drop = nn.Dropout(drop_feat)

        self.fc = nn.Linear(feats_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        self.Encoder = GCN(hidden_dim, embed_dim, dropout)
        self.att = Attention(embed_dim)
        self.decoder = nn.Sequential(
                                     nn.Linear(2*embed_dim, hidden_dim),
                                     nn.ELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, feats_dim)
                                     )
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        self.contrast_l = Contrast(embed_dim, embed_dim, act, self.tau)
        self.contrast_h = Contrast(embed_dim, embed_dim, act, self.tau)

    def forward(self, feat, adjs_l, adjs_h, adjs_o, pos):


        adj_I = torch.eye(self.nnodes, self.nnodes).to(feat.device)

        h_mp = self.act(self.fc(feat))
        h_mask = self.act(self.fc(self.drop(feat)))

        h_l_ = []
        h_h_ = []

        z_l = self.Encoder(h_mp, adjs_l[0], self.nlayer_c)
        z_h = self.Encoder(h_mp, adjs_h[0], self.nlayer_c)

        for i in range(self.sub_num):

            filter_l_o = adjs_o[i].to_dense()
            filter_h_o = adj_I - adjs_o[i].to_dense()
            h_l_.append(self.Encoder(h_mask, filter_l_o, self.nlayer))
            h_h_.append(self.Encoder(h_mask, filter_h_o, self.nlayer))

        z = self.att(h_l_, h_h_)

        loss_rec = 0
        loss_l = 0
        loss_h = 0
        for i in range(self.sub_num):
            fea_rec = self.decoder(torch.cat((h_l_[i],h_h_[i]),dim=-1))
            loss_rec += sce_loss(fea_rec, feat, self.alpha)

        loss_rec = loss_rec / self.sub_num


        loss_l += self.contrast_l(z_l, z, pos)
        loss_h += self.contrast_h(z_h, z, pos)

        loss = loss_l + loss_h + loss_rec

        return loss

    def get_embeds(self, feat, adjs_o):
        adj_I = torch.eye(self.nnodes, self.nnodes).to(feat.device)

        h_mp_l = self.act(self.fc(feat))
        h_l_ = []
        h_h_ = []
        for i in range(self.sub_num):

            filter_l_o = adjs_o[i].to_dense()
            filter_h_o = adj_I - adjs_o[i].to_dense()
            h_l_.append(self.Encoder(h_mp_l, filter_l_o, self.nlayer))
            h_h_.append(self.Encoder(h_mp_l, filter_h_o, self.nlayer))

        z = self.att(h_l_,h_h_)
        return z.detach()





class LatGRL_Sampler(nn.Module):
    def __init__(self, feats_dim, sub_num, hidden_dim, embed_dim, tau, dropout, act, drop_feat, nnodes, dataset, alpha, nlayer):
        super(LatGRL_Sampler, self).__init__()
        self.alpha = alpha
        self.feats_dim = feats_dim
        self.embed_dim = embed_dim
        self.sub_num = sub_num
        self.tau = tau
        self.dataset = dataset
        self.nnodes = nnodes
        self.act = act
        self.nlayer = nlayer
        self.nlayer_c = nlayer


        if dataset == 'mag':
            self.fc = nn.Sequential(nn.Dropout(drop_feat),
                                    nn.Linear(feats_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, embed_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, embed_dim)
                                    )
        else:
            self.fc = nn.Sequential(nn.Dropout(drop_feat),
                                    nn.Linear(feats_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, embed_dim),
                                    )

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        self.att = Attention(embed_dim, dropout)
        self.decoder = nn.Sequential(
                                     nn.Linear(2*embed_dim, hidden_dim),
                                     nn.ELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, feats_dim)
                                     )
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        self.contrast_l = Contrast(embed_dim, embed_dim, act, self.tau)
        self.contrast_h = Contrast(embed_dim, embed_dim, act, self.tau)

    def forward(self, adjs_l_filter_sampler, adjs_h_filter_sampler, lg_l_filter_sampler, lg_h_filter_sampler, feat):

        adj_I = torch.eye(len(lg_l_filter_sampler)).to_sparse().to(lg_l_filter_sampler.device)

        z_l = F.normalize(self.fc(lg_l_filter_sampler), p=2, dim=-1)
        z_h = F.normalize(self.fc(lg_h_filter_sampler), p=2, dim=-1)


        h_l_ = []
        h_h_ = []
        for i in range(self.sub_num):

            h_l_.append(F.normalize(self.fc(adjs_l_filter_sampler[i]), p=2, dim=-1))
            h_h_.append(F.normalize(self.fc(adjs_h_filter_sampler[i]), p=2, dim=-1))

        z = self.att(h_l_, h_h_)

        loss_rec = 0
        loss_l = 0
        loss_h = 0

        loss_l += self.contrast_l(z_l, z, adj_I)
        loss_h += self.contrast_h(z_h, z, adj_I)

        for i in range(self.sub_num):
            fea_rec = self.decoder(torch.cat((h_l_[i],h_h_[i]),dim=-1))
            loss_rec += sce_loss(fea_rec, feat, self.alpha)

        loss_rec = loss_rec / self.sub_num

        loss = loss_l + loss_h + loss_rec

        return loss

    def get_embeds_sampler(self, adjs_l_filter_sampler, adjs_h_filter_sampler):

        h_l_ = []
        h_h_ = []
        for i in range(self.sub_num):

            h_l_.append(F.normalize(self.fc(adjs_l_filter_sampler[i]), p=2, dim=-1))
            h_h_.append(F.normalize(self.fc(adjs_h_filter_sampler[i]), p=2, dim=-1))


        z = self.att(h_l_, h_h_)

        return z.detach()



