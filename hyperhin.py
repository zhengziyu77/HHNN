import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from models.mlp import MLP

import numpy as np
import math
from torch_geometric.utils import softmax

from torch_scatter import scatter
# NOTE: can not tell which implementation is better statistically 

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X
class HyperHINConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, in_layer_shape, out_layer_shape, node_ids, heads=8, dropout=0., negative_slope=0.2, skip_sum=False):
        super().__init__()
        
        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop  = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.args = args
        self.node_ids = node_ids
        self.reset_parameters()

        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.W_rel = nn.ParameterDict()
        for k in in_layer_shape:
            self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[k], heads * out_channels))
            nn.init.xavier_uniform_(self.W_rel[k].data, gain=1.414)

        self.P_rel = nn.ParameterDict()
        for k in in_layer_shape:
            self.P_rel[k] = nn.Parameter(torch.FloatTensor(heads * out_channels , heads * out_layer_shape[k]))
            nn.init.xavier_uniform_(self.P_rel[k].data, gain=1.414)
            

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, X, X_dict, vertex, edges):#vertex id
       
        H, C, N = self.heads, self.out_channels, X.shape[0]
        E = len(edges)
  
        # X0 = X # NOTE: reserved for skip connection

        X0 = self.W(X) #k
        X1 = X0.view(N, H, C)
        n2e_list = []
        for n_type in X_dict:
            n2e = torch.mm(X_dict[n_type], self.W_rel[n_type])
            n2e_list.append(n2e)
        X_n2e = torch.cat(n2e_list,dim=0)

        X = X_n2e.view(N, H, C)
        use_node_attn = True
        X = X1
        if use_node_attn:
            alpha_v = (X * self.att_v).sum(-1) # [V, H, 1] #QK
            a_ve = alpha_v[vertex]
            lamb = a_ve # Recommed to use this
            lamb = self.leaky_relu(lamb)
            lamb = softmax(lamb, edges, num_nodes=E)
            #lamb = self.attn_drop( lamb )
            lamb = lamb.unsqueeze(-1)
            Xve = X[vertex] # [nnz, H, C]
  
            Xve = Xve * lamb #QKV
            Xe = scatter(Xve, edges, dim=0, reduce='sum', dim_size=E) # [E, H, C]
        else:
            Xve = X[vertex] # [nnz, H, C]
            Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, H, C]

        alpha_e = (Xe * self.att_e).sum(-1) # [E, H, 1]
        a_ev = alpha_e[edges]
        alpha = a_ev # Recommed to use this
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop( alpha )
        alpha = alpha.unsqueeze(-1)


        Xev = Xe[edges] # [nnz, H, C]
        Xev = Xev * alpha 
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, H, C]
        

        #e2n_list = []
        #for n_type in X_dict:
        #    print(self.P_rel[n_type])
        #    e2n = torch.mm(Xv[self.node_ids[n_type]], self.P_rel[n_type])
        #    e2n_list.append(e2n)#存储超边表示投影到各个类型节点空间后的特征
        #X_e2n = torch.cat(e2n_list,dim=0)

        X = Xv.view(N, H * C)
        #e2n_list = []
        #for n_type in X_dict:
        #    e2n = torch.mm(X[self.node_ids[n_type]], self.P_rel[n_type])
        #    e2n_list.append(e2n)
        #X_e2n = torch.cat(e2n_list,dim=0)
        #X = X_e2n


        if self.args.use_norm:
            X = normalize_l2(X)

        if self.skip_sum:
            X = X + X0 

        # NOTE: concat heads or mean heads?
        # NOTE: skip concat here?

        return X
    
class HyperHINLayer(nn.Module):
    def __init__(self, args, in_channels, out_channels, in_layer_shape, out_layer_shape, node_ids, heads=8, dropout=0., negative_slope=0.2, hyperedge_dict={}, skip_sum=False):
        super(HyperHINLayer, self).__init__()

        self.hyperedge_dict = hyperedge_dict
        self.unigat_agg = nn.ModuleDict()
        
        for k in hyperedge_dict:
            self.unigat_agg[k] = HyperHINConv(args, in_channels, out_channels, in_layer_shape, out_layer_shape, node_ids, heads=heads, dropout=args.attn_drop)

        self.w_self = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

        self.bias = nn.Parameter(torch.FloatTensor(1, heads*out_channels))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)


        type_att_size =128
        self.w_query = nn.Parameter(torch.FloatTensor( out_channels, type_att_size))
        nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
        self.w_keys = nn.Parameter(torch.FloatTensor(heads * out_channels, type_att_size))
        nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
        self.w_att = nn.Parameter(torch.FloatTensor(2*type_att_size, 1))
        nn.init.xavier_uniform_(self.w_att.data, gain=1.414)


    def forward(self, X, X_dict):
        x_list = []
        self_x = torch.mm(X, self.w_self)
        for k in self.unigat_agg.keys():
            hyperedge = self.hyperedge_dict[k]
            V,E = hyperedge[0], hyperedge[1]
            curr_x = self.unigat_agg[k](X, X_dict,V,E)
            x_list.append(curr_x)
        #hyperedge_type fusion
        #mean

        agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in x_list], 1).mean(1)
        """
        #attn
        att_query = torch.mm(self_x, self.w_query).repeat(len(x_list), 1)
        att_keys = torch.mm(torch.cat(x_list, 0), self.w_keys)
        att_input = torch.cat([att_keys, att_query], 1)
        att_input = F.dropout(att_input, 0.5, training=self.training)
        e = F.elu(torch.matmul(att_input, self.w_att))
        attention = F.softmax(e.view(len(x_list), -1).transpose(0,1), dim=1)
        agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in x_list], 1).mul(attention.unsqueeze(-1)).sum(1)
        """
        output = agg_nb_ft# + self.bias

        return output


class HyperHIN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, nhead, in_layer_shape, hid_layer_shape, out_layer_shape, node_ids, hyperedge_dict, args):
        """UniGNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.conv_out = HyperHINLayer(args, nhid * nhead, nclass, hid_layer_shape, out_layer_shape, node_ids, heads=1, dropout=args.attn_drop,hyperedge_dict = hyperedge_dict)
        self.convs = nn.ModuleList(
            [ HyperHINLayer(args, nfeat, nhid, in_layer_shape, hid_layer_shape, node_ids, heads=nhead, dropout=args.attn_drop, hyperedge_dict = hyperedge_dict)] +
            [HyperHINLayer(args, nhid * nhead, nhid, heads=nhead, dropout=args.attn_drop, hyperedge_dict = hyperedge_dict) for _ in range(nlayer-2)]
        )
        #self.V = V 
        #self.E = E 
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, data, X_dict):
        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]
        X = self.input_drop(x)
        for conv in self.convs:
            X = conv(X, X_dict)
            X = self.act(X)
            X = self.dropout(X)

        X = self.conv_out(X, X_dict)      
        return F.log_softmax(X, dim=1)


__all_convs__ = {
    'HyperHIN': HyperHINConv,
}
class UniGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, nhead,args):
        """UniGNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        Conv = __all_convs__['UniGAT']
        self.conv_out = Conv(args, nhid * nhead, nclass, heads=1, dropout=args.attn_drop)
        self.convs = nn.ModuleList(
            [ Conv(args, nfeat, nhid, heads=nhead, dropout=args.attn_drop)] +
            [Conv(args, nhid * nhead, nhid, heads=nhead, dropout=args.attn_drop) for _ in range(nlayer-2)]
        )
        #self.V = V 
        #self.E = E 
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, data):

        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]
        X = self.input_drop(x)
        for conv in self.convs:
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)

        X = self.conv_out(X, V, E)      
        return F.log_softmax(X, dim=1)
