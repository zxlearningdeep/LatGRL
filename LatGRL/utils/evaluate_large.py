import numpy as np
import torch
from .mlp_classifier import MLP
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
# from torch_kmeans import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import RandomSampler
import math


def evaluate_large(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , starttime, epoch=0, isTest=True):
    torch.cuda.empty_cache()
    num_target_node_train = len(idx_train)
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    embeds = embeds.cpu()
    label = label.cpu()
    idx_train = idx_train.cpu()
    idx_val = idx_val.cpu()
    idx_test = idx_test.cpu()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1).long()
    print(train_lbls)
    print(val_lbls.max())
    print(val_lbls)
    test_lbls = torch.argmax(label[idx_test], dim=-1).long()
    accs = []
    accs_val = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []

    for turn in range(10):
        sampler = RandomSampler(range(num_target_node_train), replacement=False)
        sampler = [i for i in sampler]
        sampler = torch.tensor(sampler)


        batchsize = 5000
        train_sampler_num = math.ceil(num_target_node_train / batchsize)

        log = MLP(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)


        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        for iter_ in range(50):
            # train
            torch.cuda.empty_cache()
            log.train()
            loss_train_epoch = 0
            train_accs = []
            for i in range(train_sampler_num):
                seed_node = torch.tensor(range(i * batchsize, (i + 1) * batchsize))
                if (i + 1) * batchsize > num_target_node_train:
                    seed_node = torch.tensor(range(i * batchsize, num_target_node_train))
                seed_node = sampler[seed_node]

                embeds_sampler = train_embs[seed_node].cuda()
                labels_sampler = train_lbls[seed_node].cuda()

                opt.zero_grad()
                logits = log(embeds_sampler)

                preds = torch.argmax(logits.cpu(), dim=1)
                acc_num = torch.sum(preds == labels_sampler.cpu()).float()
                train_accs.append(acc_num)

                loss = xent(logits, labels_sampler)
                loss.backward()
                opt.step()

                loss_train_epoch += loss.item()
                # print("Epoch:", iter_, "Batch:", i, "loss:", loss.item())

            loss_train_epoch = loss_train_epoch / train_sampler_num
            train_acc = sum(train_accs) / num_target_node_train
            print('Train loss: ', loss_train_epoch, 'Train ACC: ', train_acc)

            # val
            torch.cuda.empty_cache()
            log.eval()
            val_preds = []
            val_sampler_num = math.ceil(len(idx_val) / batchsize)
            for i in range(val_sampler_num):
                seed_node = torch.tensor(range(i * batchsize, (i + 1) * batchsize))
                if (i + 1) * batchsize > len(idx_val):
                    seed_node = torch.tensor(range(i * batchsize, len(idx_val)))

                embeds_sampler = val_embs[seed_node].cuda()

                logits = log(embeds_sampler)
                preds = torch.argmax(logits.cpu(), dim=1)
                val_preds.append(preds)

            val_preds = torch.cat(val_preds, dim=0)
            val_acc = torch.sum(val_preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), val_preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), val_preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)


            # test
            torch.cuda.empty_cache()
            log.eval()
            test_preds = []
            test_sampler_num = math.ceil(len(idx_test) / batchsize)
            for i in range(test_sampler_num):
                seed_node = torch.tensor(range(i * batchsize, (i + 1) * batchsize))
                if (i + 1) * batchsize > len(idx_test):
                    seed_node = torch.tensor(range(i * batchsize, len(idx_test)))

                embeds_sampler = test_embs[seed_node].cuda()

                logits = log(embeds_sampler)
                preds = torch.argmax(logits.cpu(), dim=1)
                test_preds.append(preds)


            test_preds = torch.cat(test_preds, dim=0)
            test_acc = torch.sum(test_preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), test_preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), test_preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

            print("\tTurn: ", turn, " Epoch: ", iter_, " val acc: ", val_acc, " test acc: ", test_acc, "ma-f1: ", test_f1_macro, "mi-f1: ", test_f1_micro)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        accs_val.append(val_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        print(
            "\t[Test Classification] Val-Acc: {:.2f} var: {:.2f}  Test-Acc: {:.2f} var: {:.2f}  Macro-F1_mean: {:.2f} var: {:.2f}  Micro-F1_mean: {:.2f} var: {:.2f}"
            .format(np.mean(accs_val) * 100,
                    np.std(accs_val) * 100,
                    np.mean(accs) * 100,
                    np.std(accs) * 100,
                    np.mean(macro_f1s) * 100,
                    np.std(macro_f1s) * 100,
                    np.mean(micro_f1s) * 100,
                    np.std(micro_f1s) * 100
                    )
            )

    if isTest:
        print(
            "\t[Test Classification] Val-Acc: {:.2f} var: {:.2f}  Test-Acc: {:.2f} var: {:.2f}  Macro-F1_mean: {:.2f} var: {:.2f}  Micro-F1_mean: {:.2f} var: {:.2f}"
            .format(np.mean(accs_val) * 100,
                    np.std(accs_val) * 100,
                    np.mean(accs) * 100,
                    np.std(accs) * 100,
                    np.mean(macro_f1s) * 100,
                    np.std(macro_f1s) * 100,
                    np.mean(micro_f1s) * 100,
                    np.std(micro_f1s) * 100
                    )
            )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    f = open("result_" + dataset + str(ratio) + ".txt", "a")
    f.write("\n"+str(starttime.strftime('%Y-%m-%d %H:%M')) + "\t")
    f.write("Epoch: " + str(epoch) + "\t")
    f.write("Val-Acc_mean: " + str(np.around(np.mean(accs_val) * 100, 2)) + " +/- " + str(
        np.around(np.std(accs_val) * 100, 2)) + "\t")
    f.write(" Test-Acc_mean: " + str(np.around(np.mean(accs) * 100, 2)) + " +/- " + str(
        np.around(np.std(accs) * 100, 2)) + "\t")
    f.write(" Ma-F1_mean: " + str(np.around(np.mean(macro_f1s) * 100, 2)) + " +/- " + str(
        np.around(np.std(macro_f1s) * 100, 2)) + "\t")
    f.write(" Mi-F1_mean: " + str(np.around(np.mean(micro_f1s) * 100, 2)) + " +/- " + str(
        np.around(np.std(micro_f1s) * 100, 2)) + "\n")
    f.close()

    return np.mean(macro_f1s) * 100, np.mean(micro_f1s) * 100


