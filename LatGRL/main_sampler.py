import numpy as np
import torch
from utils import load_data, set_params_large, evaluate, run_kmeans,evaluate_large
from module.LatGRL import *
from module.graph_sampler_generating import *
from module.preprocess import *
import warnings
import datetime
import time
import pickle as pkl
import random
import matplotlib.pyplot as plt
import os
import math

warnings.filterwarnings('ignore')
args = set_params_large()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")


## name of intermediate document ##
own_str = args.dataset

## random seed ##
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train():
    if args.dataset == 'mag':
        feat, adjs_l_filter, adjs_h_filter, lg_l_filter, lg_h_filter, label, idx_train, idx_val, idx_test = \
            load_data(args.dataset, args.ratio, args.type_num)

        nb_classes = label.shape[-1]
        num_target_node = len(feat)
        feats_dim = feat.shape[1]
        sub_num = int(len(adjs_l_filter))
    else:
        feat, adjs, label, idx_train, idx_val, idx_test = \
            load_data(args.dataset, args.ratio, args.type_num)
        adjs_l, adjs_h, adjs_o = graph_process_sampler(adjs, feat, args)
        adjs_l_filter, adjs_h_filter, lg_l_filter, lg_h_filter = pre_filter(feat, adjs_o, adjs_l[0], adjs_h[0],
                                                                            args.nlayer, device)

        nb_classes = label.shape[-1]
        num_target_node = len(feat)

        feats_dim = feat.shape[1]
        sub_num = int(len(adjs))
        print("Dataset: ", args.dataset)
        print("The number of meta-paths: ", sub_num)
        print("Number of target nodes:", num_target_node)
        print("The dim of target' nodes' feature: ", feats_dim)
        print("Label: ", label.sum(dim=0))
        print(args)


    sampler_fun = Node_Sampler(adjs_l_filter, adjs_h_filter, lg_l_filter, lg_h_filter, feat, num_target_node, args.batchsize)

    model = LatGRL_Sampler(feats_dim, sub_num, args.hidden_dim, args.embed_dim, args.tau, args.dropout, args.act, args.drop_feat, num_target_node, args.dataset, args.alpha, args.nlayer)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        model.cuda()

    cnt_wait = 0
    best = 1e9
    period = 40
    best_epoch=0
    batch_num = math.ceil(num_target_node / args.batchsize)

    starttime = datetime.datetime.now()

    if not args.load_parameters:
        print("start training")
        t0 = time.time()
        for epoch in range(args.nb_epochs):

            model.train()
            loss_epoch = 0
            sampler_fun.random_sampler()
            for i in range(batch_num):
                adjs_l_filter_sampler, adjs_h_filter_sampler, lg_l_filter_sampler, lg_h_filter_sampler, feat_sampler = sampler_fun.filter_sampler(i)
                optimizer.zero_grad()
                loss = model(adjs_l_filter_sampler, adjs_h_filter_sampler, lg_l_filter_sampler, lg_h_filter_sampler, feat_sampler)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()

            loss_epoch = loss_epoch / batch_num
            print('Epoch: ', epoch, 'Total loss: ', loss_epoch)

            if best > loss_epoch:
                best = loss_epoch
                cnt_wait = 0
                best_epoch = epoch
                print('save checkpoint!')
                torch.save(model.state_dict(), './checkpoint/'+args.dataset+'/sampler/best_'+str(args.seed)+'.pth')
            else:
                cnt_wait += 1
            if cnt_wait >= args.patience:
                break

        t1 = time.time()
        training_time = t1 - t0
        training_time = format_time(training_time)
        print("Training Time:", training_time)

        model.load_state_dict(torch.load('./checkpoint/' + args.dataset + '/sampler/best_' + str(args.seed) + '.pth'))
    else:
        model.load_state_dict(torch.load('./best/sampler/' + args.dataset + '/best_' + str(0) + '.pth'))
    model.cuda()
    epoch = best_epoch
    model.eval()



    adjs_l_filter_sampler, adjs_h_filter_sampler = sampler_fun.filter_sampler_test(0)
    embeds = model.get_embeds_sampler(adjs_l_filter_sampler, adjs_h_filter_sampler).cpu()
    for i in range(1, batch_num):
        adjs_l_filter_sampler, adjs_h_filter_sampler = sampler_fun.filter_sampler_test(i)
        embeds_ = model.get_embeds_sampler(adjs_l_filter_sampler, adjs_h_filter_sampler).cpu()
        embeds = torch.cat((embeds, embeds_), dim=0)


    model.cpu()
    if args.dataset == 'mag':
        for i in range(len(idx_train)):
            ma, mi = evaluate_large(embeds, args.ratio[i], idx_train[i].cuda(), idx_val[i].cuda(), idx_test[i].cuda(), label.cuda(), nb_classes, device,
                                   args.dataset,
                                   args.eva_lr, args.eva_wd, starttime)
    else:
        embeds = embeds.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

        nmi, ari = run_kmeans(embeds.cpu(), torch.argmax(label.cpu(), dim=-1), nb_classes, starttime, args.dataset,
                              epoch + 1)
        for i in range(len(idx_train)):
            ma, mi, auc = evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes,
                                   device,
                                   args.dataset,
                                   args.eva_lr, args.eva_wd, starttime, epoch + 1)




if __name__ == '__main__':
    set_seed(args.seed)
    train()


