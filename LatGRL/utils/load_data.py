import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return th.sparse.FloatTensor(index, th.ones(len(index[0])), adj.shape)

def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def sp_tensor_to_sp_csr(adj):
    adj = adj.coalesce()
    row = adj.indices()[0]
    col = adj.indices()[1]
    data = adj.values()
    shape = adj.size()
    adj = sp.csr_matrix((data, (row, col)), shape=shape)
    return adj



def load_acm(ratio):

    path = "./data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_p = sp.load_npz(path + "p_feat.npz")

    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    adjs = [pap, psp]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj).coalesce() for adj in adjs]


    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)

    feat_p = th.FloatTensor(preprocess_features(feat_p))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return feat_p, adjs, label, train, val, test


def load_dblp(ratio):
    path = "./data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")

    apa = sp.load_npz(path + "apa.npz")  
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    adjs = [apa, apcpa, aptpa]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj)for adj in adjs]

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)
    feat_a = th.FloatTensor(preprocess_features(feat_a))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return feat_a, adjs, label, train, val, test


def load_imdb(ratio):

    # data obtained from pyg imdb
    path = "./data/imdb/"
    x = np.load(path+'features_movie.npy')
    x_sum = x.sum(axis=1)
    x_ = np.power(x_sum, -1)
    x_[np.isinf(x_)] = 0.
    x_ = np.diag(x_)
    x = x_ @ x
    x = th.FloatTensor(x)

    label = np.load(path + "label.npy").astype('int32')
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    mdm = sp.load_npz(path+'mdm_.npz')
    mam = sp.load_npz(path+'mam_.npz')

    adjs = [mdm, mam]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return x, adjs, label, train, val, test


def load_yelp(ratio):

    path = "./data/yelp/"
    feat_b = sp.load_npz(path + "features_0.npz").astype("float32")
    feat_b = th.FloatTensor(preprocess_features(feat_b))
    label = np.load(path+'labels.npy')
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    blb = np.load(path+'blb.npy').astype("float32")
    bsb = np.load(path+'bsb.npy').astype("float32")
    bub = np.load(path+'bub.npy').astype("float32")


    adjs = [bsb, bub, blb]
    adjs = [th.tensor(adj).to_sparse() for adj in adjs]

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]


    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return feat_b, adjs, label, train, val, test


def load_mag(flag):

    path = "./data/mag/"
    path_ = "./data/mag/pre_filter/"

    feat_p = th.load(path+'paper_features.pt')
    mp2vec_emb = torch.load(path+'paper_mp2vec.pt')
    feat_p = torch.cat((feat_p, mp2vec_emb), dim=1)

    adjs_l_filter = torch.load(path_+'adjs_l_filter.pt')
    adjs_h_filter = torch.load(path_+'adjs_h_filter.pt')
    lg_l_filter = torch.load(path_+'lg_l_filter.pt')
    lg_h_filter = torch.load(path_+'lg_h_filter.pt')

    label = th.load(path + 'label.pt').numpy()
    label = encode_onehot(label)
    label = th.FloatTensor(label)
    label_ = th.load(path + 'label.pt')

    train_idx = th.load(path + 'train_idx.pt')
    val_idx = th.load(path + 'val_idx.pt')
    test_idx = th.load(path + 'test_idx.pt')
    print(label_[test_idx].unique().shape, label_[test_idx].unique().shape)

    return feat_p, adjs_l_filter, adjs_h_filter, lg_l_filter, lg_h_filter, label, [train_idx], [val_idx], [test_idx]






def load_data(dataset, ratio, type_num, flag=None):
    if dataset == "acm":
        data = load_acm(ratio)
    elif dataset == "dblp":
        data = load_dblp(ratio)
    elif dataset == 'imdb':
        data = load_imdb(ratio)
    elif dataset == 'yelp':
        data = load_yelp(ratio)
    elif dataset == 'mag':
        data = load_mag(flag)

    return data

