

import torch
import pickle
import os
#import ipdb
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_dense_adj
from torch_sparse import coalesce
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz, load_npz
import numpy as np
import os.path as osp
from load_heterdata import *


import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

def convert_H_to_hyperedge_index(H):
    """
    Convert a hypergraph incidence matrix H to hyperedge_index format using PyTorch without explicit iteration.
    :param H: Hypergraph incidence matrix H as a PyTorch tensor.
    :return: hyperedge_index as a PyTorch tensor.
    """
    # 确保H是PyTorch张量
    H = H.T
    
    # 找到非零元素的索引，其中每一列的索引代表一条超边中的节点
    hyperedge,node_id = torch.nonzero(H, as_tuple=True)
    
    # 将索引转换为超边索引格式，其中每个超边的索引放在同一行
    hyperedge_index = torch.stack((node_id,hyperedge), dim=1).t()

    return hyperedge_index

def init_feat(num_list, n_inp):
    # Randomly initialize features if features don't exist
    x = []
    X_dict = {}
    for k in num_list:
        emb = torch.nn.Parameter(torch.Tensor(num_list[k], n_inp[k]), requires_grad=True)
        torch.nn.init.xavier_uniform_(emb,gain=1.414)
        emb = scale_feats(emb.detach())
        x.append(emb)
        X_dict[k] = emb
    X = torch.cat(x,dim=0)

    #x = torch.nn.Parameter(torch.Tensor(78515,30), requires_grad=True)
    #torch.nn.init.xavier_uniform_(x,gain=1.414)

    return X, X_dict
def split_dataset(X, train_size=0.6, val_size=0.2, test_size=0.2):
    # Create a random permutation of the indices of the samples
    indices = np.random.permutation(X.shape[0])

    # Split the indices into the desired number of sets
    train_indices = indices[:int(train_size * X.shape[0])]
    val_indices = indices[int(train_size * X.shape[0]):int((train_size + val_size) * X.shape[0])]
    test_indices = indices[int((train_size + val_size) * X.shape[0]):]

    return train_indices, val_indices, test_indices
    
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask
def scale_feats(x):
    scaler = StandardScaler()
    #scaler = MaxAbsScaler()
    
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
def get_input_dim(X_dict):
    input_layer_shape={}
    for k in X_dict:
        input_layer_shape[k] = X_dict[k].shape[1]
    return input_layer_shape
def load_moivelens():

    H = torch.load('/Heterogeneous_Hyper_Data/HGNN/clean_data/incidence_matrix.pt')
    H_tag= torch.load('/Heterogeneous_Hyper_Data/HGNN/clean_data/tag_incidence_matrix.pt')
    H_rate = torch.load('/Heterogeneous_Hyper_Data/HGNN/clean_data/rate_incidence_matrix.pt')

    labels = torch.load('/Heterogeneous_Hyper_Data/HGNN/clean_data/movies_labels.pt')

    #H = torch.load('./data/incidence_matrix.pt')
    #labels = torch.load('./data/movies_labels.pt')

    num_list = {'movie': 3439, 'tag': 3108, 'user': 2106, 'rate':10} #8663,168578
    num_list = {'m': 3439, 't': 3108, 'u': 2106, 'r':10} #8663,168578


    hyperedge_dict={}
    hyperedge_index = convert_H_to_hyperedge_index(H)
    hyperedge_index1 = convert_H_to_hyperedge_index(H_tag)
    hyperedge_index2 = convert_H_to_hyperedge_index(H_rate)

    hyperedge_dict['umt'] = hyperedge_index1
    hyperedge_dict['umr'] = hyperedge_index2

    init_dims =  {'m': 128, 't': 128, 'u': 128, 'r':128}


    feat, X_dict = init_feat(num_list, init_dims)

    data = Data(x=feat, x_dict=X_dict, edge_index=hyperedge_index, y=labels, num_features=init_dims, num_classes=8, num_nodes=8663, num_hyperedges=168578)

    # 初始化起始ID
    start_id = 0
    adj = [H,H_tag,H_rate]
    data.adj =adj

    # 创建一个空字典来存储节点ID
    node_ids = {}

    # 为每种类型的节点生成ID列表
    for node_type in num_list:
        node_ids[node_type] = list(range(start_id, start_id + num_list[node_type]))
        start_id += num_list[node_type]  # 更新下一个节点类型的起始ID

    
    target_type = 'movie'
    hyper_list = ['umt', 'umr']
    target_type = 'm'
    return data, hyperedge_dict, node_ids, target_type, hyper_list


def load_customer():

    labels = torch.load('./olist/processed/clean_data/product_labels.pt')
    seller_hyperedge = torch.load('./olist/processed/clean_data/seller_hyperedge.pt')
    rate_hyperedge = torch.load('./olist/processed/clean_data/rate_hyperedge.pt')
    price_hyperedge = torch.load('./olist/processed/clean_data/price_hyperedge.pt')
    hyperedge_index = torch.load('./olist/processed/clean_data/hyperedge.pt')
    #H = torch.load('./data/incidence_matrix.pt')
    #labels = torch.load('./data/movies_labels.pt')

    #num_list = [3557, 3140, 2106, 10] 8813,170780
    num_list = {'product': 18454, 'customer':54321, 'seller':2090, 'location':3540, 'price':14, "rate":5}
    num_list = {'p': 18454, 'c':54321, 's':2090, 'd':3540,  "r":5,'v':14}

    hyperedge_dict={}
  

    hyperedge_dict["cps"] = seller_hyperedge
    hyperedge_dict["cpr"] = rate_hyperedge
    hyperedge_dict["dpv"] = price_hyperedge
    #hyperedge_dict['total'] = hyperedge_index

    #hyperedge_dict['total'] = hyperedge_index
    init_dims = {'product': 30, 'customer':30, 'seller':30, 'location':30, "rate":30, 'price':30}
    #init_dims = {'p': 30, 'c':30, 's':30, 'd':30, 'r':30, "v":30}
    init_dims = {'p': 128, 'c':128, 's':128, 'd':128, 'r':128, "v":128}

    #init_dims =  {'movie': 30, 'tag': 30, 'user': 30, 'rate':30}

    #init_dims =  {'movie': 128, 'tag': 128, 'user': 128, 'rate':128}
    #num_list = {'p': 18454, 'c':54321, 's':2090}
    #init_dims = {'p': 128, 'c':128, 's':128}


    feat, X_dict = init_feat(num_list, init_dims)
   


    data = Data(x=feat, x_dict=X_dict, edge_index=hyperedge_index, y=labels, num_features=init_dims, num_classes=10, num_nodes=78424, num_hyperedges=166816)
    
    cps = torch.load('/olist/processed/clean_data/cps.pt')
    cpr = torch.load('/olist/processed/clean_data/cpr.pt')
    spd = torch.load('/olist/processed/clean_data/spd.pt')

    adj = [cps, cpr, spd]
    data.adj =adj
    # 初始化起始ID
    start_id = 0

    # 创建一个空字典来存储节点ID
    node_ids = {}

    # 为每种类型的节点生成ID列表
    for node_type in num_list:
        node_ids[node_type] = list(range(start_id, start_id + num_list[node_type]))
        start_id += num_list[node_type]  # 更新下一个节点类型的起始ID
    target_type ='product'
    target_type = 'p'
    hyper_list = ['cps', 'cpr', 'dpv']

    return data, hyperedge_dict, node_ids, target_type, hyper_list






import numpy as np
from collections import defaultdict
import pickle
import torch as th
import torch.nn.functional as F

import scipy.sparse as sp
import process
import os.path as osp
np.random.seed(0)



def load_homogeneous_graph(name):
    if name == 'movielens':
        data, hyperedge_dict, node_ids, target_type, hyperedge_list = load_moivelens()
    elif name == 'olist':
        data, hyperedge_dict, node_ids, target_type, hyperedge_list = load_customer()
    return data, hyperedge_dict, node_ids, target_type, hyperedge_list




