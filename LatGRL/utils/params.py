import argparse
import torch
import sys

argv = sys.argv
dataset = argv[1]
# dataset = 'acm'


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=500)  # 400
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=0)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_feat', type=float, default=0.1)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--graph_k', type=int, default=3)
    parser.add_argument('--k_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=2)

    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=97)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-4)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--drop_feat', type=float, default=0.1)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--graph_k', type=int, default=10)
    parser.add_argument('--k_pos', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=2)

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    return args



def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=60)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.03)
    parser.add_argument('--eva_wd', type=float, default=1e-4)

    # The parameters of learning process
    # parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_feat', type=float, default=0.6)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--graph_k', type=int, default=3)
    parser.add_argument('--k_pos', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)

    args, _ = parser.parse_known_args()
    args.type_num = [4278, 2081, 5257]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args




def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=60)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    # parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_feat', type=float, default=0.2)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--graph_k', type=int, default=10)
    parser.add_argument('--k_pos', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=2)

    args, _ = parser.parse_known_args()
    args.type_num = [2614, 1286, 4, 9]  # the number of every node type
    args.nei_num = 4  # the number of neighbors' types
    return args



def acm_sampler_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--LG_construction', type=str, default=False)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=0)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--drop_feat', type=float, default=0.2)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--graph_k', type=int, default=3)
    parser.add_argument('--k_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--anchor_num', type=int, default=1000)

    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    return args


def dblp_sampler_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--LG_construction', type=str, default=False)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=97)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--drop_feat', type=float, default=0.)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--graph_k', type=int, default=5)
    parser.add_argument('--k_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--anchor_num', type=int, default=1000)


    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    return args


def imdb_sampler_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--LG_construction', type=str, default=False)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=100)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=60)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-4)

    # The parameters of learning process
    # parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_feat', type=float, default=0.6)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--graph_k', type=int, default=3)
    parser.add_argument('--k_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--anchor_num', type=int, default=1000)

    args, _ = parser.parse_known_args()
    args.type_num = [4278, 2081, 5257]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args



def yelp_sampler_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--LG_construction', type=str, default=False)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=500)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=60)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--drop_feat', type=float, default=0.)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1.0)  # 0.8
    parser.add_argument('--graph_k', type=int, default=10)
    parser.add_argument('--k_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--anchor_num', type=int, default=1000)

    args, _ = parser.parse_known_args()
    args.type_num = [2614, 1286, 4, 9]  # the number of every node type
    args.nei_num = 4  # the number of neighbors' types
    return args


def mag_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--LG_construction', type=str, default=False)
    parser.add_argument('--dataset', type=str, default="mag")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--nb_epochs', type=int, default=5)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=60)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=1)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.0005)
    parser.add_argument('--eva_wd', type=float, default=1e-4)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--drop_feat', type=float, default=0.)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--graph_k', type=int, default=5)
    parser.add_argument('--k_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--batchsize', type=int, default=5120)
    parser.add_argument('--anchor_num', type=int, default=5000)


    args, _ = parser.parse_known_args()
    args.type_num = [736389, 1134649, 8740, 59965]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args



def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == 'yelp':
        args = yelp_params()
    elif dataset == 'imdb':
        args = imdb_params()
    return args

def set_params_large():
    if dataset == "mag":
        args = mag_params()
    elif dataset == "acm":
        args = acm_sampler_params()
    elif dataset == "dblp":
        args = dblp_sampler_params()
    elif dataset == 'yelp':
        args = yelp_sampler_params()
    elif dataset == 'imdb':
        args = imdb_sampler_params()
    return args


