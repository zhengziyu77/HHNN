import os, sys
import math, time, random
import pickle
import argparse, configargparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch.nn.functional import softmax
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from tqdm import tqdm

from models import SetGNN, HCHA, HNHN, HyperGCN, HyperSAGE, \
    LEGCN, UniGCNII, HyperND, EquivSetGNN, UniGAT, HyperHIN
from sklearn.metrics import (average_precision_score,
                             roc_auc_score,
                             f1_score,
                             normalized_mutual_info_score,
                             adjusted_rand_score,
                             accuracy_score)
import utils
from data_utils import *
from utils import *
def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]


@torch.no_grad()
def evaluate(model, data, X_dict, split_idx, evaluator, loss_fn=None, return_out=False, best_val=None, best_test=None):
    model.eval()
    out,_ = model(data,X_dict)


    out = F.log_softmax(out, dim=1)

    train_acc = evaluator.eval(data.y[split_idx['train']], out[split_idx['train']])['acc']
    valid_acc = evaluator.eval(data.y[split_idx['valid']], out[split_idx['valid']])['acc']
    test_acc = evaluator.eval(data.y[split_idx['test']], out[split_idx['test']])['acc']

    y_pred = out[split_idx['test']].argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    micro = f1_score(y_pred, data.y[split_idx['test']].detach().cpu().numpy(), average='micro')
    macro = f1_score(y_pred, data.y[split_idx['test']].detach().cpu().numpy(), average='macro')

    y_prob = softmax(out[split_idx['test']],dim=1)
    auc = roc_auc_score(data.y[split_idx['test']].detach().cpu().numpy(), y_prob.detach().cpu().numpy(), multi_class='ovr')
    #auc =0 
  


    ret_list = [train_acc, valid_acc, test_acc, micro, macro, auc]

    # Also keep track of losses
    if loss_fn is not None:
        train_loss = loss_fn(out[split_idx['train']], data.y[split_idx['train']])
        valid_loss = loss_fn(out[split_idx['valid']], data.y[split_idx['valid']])
        test_loss = loss_fn(out[split_idx['test']], data.y[split_idx['test']])
        ret_list += [train_loss, valid_loss, test_loss]

    if return_out:
        ret_list.append(out)

    return ret_list, best_val, best_test

            
def main(args):

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')


    #data, hyperedge_dict, node_ids, target_type, hyperedge_list = load_moivelens()
    data, hyperedge_dict, node_ids, target_type, hyperedge_list = load_customer()


    # 创建值到键的映射
    value_to_key_map = {}
    for key, values in node_ids.items():
        for value in values:
            if value not in value_to_key_map:
                value_to_key_map[value] = key
    x_dict = data.x_dict
    label=data.y+1
    num_class = max(label)
    print('num_class:', num_class)
    in_layer_shape = get_input_dim(x_dict)
    print(in_layer_shape)

    output_layer_shape = dict.fromkeys(x_dict.keys(), num_class)
    data.y = data.y.to(device)
    for k in x_dict:
        x_dict[k] = x_dict[k].to(device)
    #data = load_twitter_data()
    data.edge_index = data.edge_index.to(device)

    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = SetGNN.norm_contruction(data, option=args.normtype)
    elif args.method == 'HNHN':
        data = HNHN.generate_norm(data, args)
    elif args.method == 'HyperSAGE':
        data = HyperSAGE.generate_hyperedge_dict(data)
    elif args.method == 'LEGCN':
        data = LEGCN.line_expansion(data)


    # Get splits
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = utils.rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=(1-args.train_prop)/2)
        #split_idx['train'] = idx_train
        #split_idx['valid'] = idx_val
        #split_idx['test'] = idx_test
       
        split_idx_lst.append(split_idx)

    if args.method == 'AllSetTransformer':
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method == 'AllDeepSets':
        args.AllSet_PMA = False
        args.aggregate = 'add'
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=data.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=data.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)
        data.edge_index = clique_expansion(data.edge_index)
        data.norm = torch.ones_like(data.edge_index[0],dtype=torch.float32)
        data.edge_index, data.norm = gcn_norm(data.edge_index, data.norm, add_self_loops=True)
        data.edge_index = data.edge_index.to(device)
        data.norm = data.norm.to(device)

    elif args.method in 'HCHA':
        model = HCHA(data.num_features, data.num_classes, args)
    elif args.method in 'HGNN':
        model = HGNN(data.num_features, n_class=data.num_classes, n_hid=args.MLP_hidden)
    elif args.method in 'HNHN':
        model = HNHN(data.num_features, data.num_classes, args)
    elif args.method in 'HyperGCN':
        model = HyperGCN(data.num_features, data.num_classes, args)
    elif args.method == 'HyperSAGE':
        model = HyperSAGE(data.num_features, data.num_classes, args)
    elif args.method == 'LEGCN':
        model = LEGCN(data.num_features, data.num_classes, args)
    elif args.method == 'UniGCNII':
        model = UniGCNII(data.num_features, data.num_classes, args)
    elif args.method == 'UniGAT':
        model = UniGAT(data.num_features, 128, data.num_classes, 2, 4, args)
    elif args.method == 'HyperND':
        model = HyperND(data.num_features, data.num_classes, args)
    elif args.method == 'EDGNN':
        model = EquivSetGNN(data.num_features, data.num_classes, args)
    else:
        raise ValueError(f'Undefined model name: {args.method}')
    model = model.to(device)
    print("# Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    logger = utils.Logger(args.runs, args)
    
    loss_fn = nn.NLLLoss()
    evaluator = utils.NodeClsEvaluator()

    runtime_list = []
    for run in range(args.runs):
        for k in hyperedge_dict:
            hyperedge_dict[k] = hyperedge_dict[k].to(device)
        hid_dim=128
        hid_layer_shape = dict.fromkeys(x_dict.keys(), hid_dim)
        model = HyperHIN(128, hid_dim, data.num_classes, 2, 4, in_layer_shape, hid_layer_shape, output_layer_shape, node_ids, hyperedge_dict,args)

        model = model.to(device)

        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)

        #model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

        best_val = float('-inf')
        best_test = float('-inf')
        cnt_wait = 0
        best = 1e9
        period = 100
        best_epoch=0

        for epoch in range(args.epochs):
            # Training loop
            model.train()
            optimizer.zero_grad()
            out, attention_layer = model(data, x_dict)
            #out= model(data, x_dict)


            
            


            out = F.log_softmax(out, dim=1)[node_ids[target_type]]

            loss = loss_fn(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

            if best > loss.item():
                best = loss.item()
                cnt_wait = 0
                best_epoch = epoch
                torch.save(model.state_dict(), './checkpoint/''best_'+str(args.seed)+'.pth')
            else:
                cnt_wait += 1
            if cnt_wait >= 10:
                break

        print(best_epoch)
        model.load_state_dict(torch.load('./checkpoint/'+'best_'+str(args.seed)+'.pth'))

        # Evaluation and logging
        result,best_val, best_test = evaluate(model, data, x_dict,split_idx, evaluator, loss_fn, best_val, best_test)
        logger.add_result(run, *result[:6])

        """
        logger.add_result(run, *result[:6])
        if epoch % args.display_step == 0 and args.display_step > 0:

            #result, best_val, best_test = evaluate(model, data, x_dict,split_idx, evaluator, loss_fn, best_val, best_test)
            #logger.add_result(run, *result[:3])

            print("run:{}, epoch:{}, loss:{:.4f}, Test Acc:{:.2f}, Micro:{:.2f}, Macro:{:.2f}, AUC:{:.2f}, ".format(run, epoch, loss, 100 * result[2],\
                100 * result[3], 100 * result[4], 100 * result[5]))
    
        """

        end_time = time.time()
        runtime_list.append(end_time - start_time)

    logger.print_statistics()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', is_config_file=True)

    # Dataset specific arguments
    parser.add_argument('--dname', default='walmart-trips-100')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.4)
    parser.add_argument('--feature_noise', default='1', type=str, help='std for synthetic feature noise')
    parser.add_argument('--normtype', default='all_one', choices=['all_one','deg_half_sym'])
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--exclude_self', action='store_true', help='whether the he contain self node or not')

    # Training specific hyperparameters
    parser.add_argument('--epochs', default=300, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=10, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--input_dropout', default=0.2, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--display_step', type=int, default=50)

    # Model common hyperparameters
    parser.add_argument('--method', default='EDGNN', help='model type')
    parser.add_argument('--All_num_layers', default=2, type=int, help='number of basic blocks')
    parser.add_argument('--MLP_num_layers', default=2, type=int, help='layer number of mlps')
    parser.add_argument('--MLP_hidden', default=64, type=int, help='hidden dimension of mlps')
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normalization', default='ln', choices=['bn','ln','None'])
    parser.add_argument('--activation', default='prelu', choices=['Id','relu', 'prelu', 'elu'])
    
    # Args for EDGNN
    parser.add_argument('--MLP2_num_layers', default=-1, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=-1, type=int, help='layer number of mlp3')
    parser.add_argument('--edconv_type', default='EquivSet', type=str, choices=['EquivSet', 'JumpLink', 'MeanDeg', 'Attn', 'TwoSets'])
    parser.add_argument('--restart_alpha', default=0.5, type=float)

    # Args for AllSet
    parser.add_argument('--AllSet_input_norm', default=True)
    parser.add_argument('--AllSet_GPR', action='store_false')  # skip all but last dec
    parser.add_argument('--AllSet_LearnMask', action='store_false')
    parser.add_argument('--AllSet_PMA', action='store_true')
    parser.add_argument('--AllSet_num_heads', default=1, type=int)
    # Args for CEGAT
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    # Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    # Args for HyperSAGE
    parser.add_argument('--HyperSAGE_power', default=1., type=float)
    parser.add_argument('--HyperSAGE_num_sample', default=100, type=int)
    # Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    # Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    # Args for UniGNN
    parser.add_argument('--UniGNN_use_norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    # Args for HyperND
    parser.add_argument('--HyperND_ord', default = 1., type=float)
    parser.add_argument('--HyperND_tol', default = 1e-4, type=float)
    parser.add_argument('--HyperND_steps', default = 100, type=int)
    parser.add_argument('--attn_drop', default =0.5, type=float)
    parser.add_argument('--input_drop', default = 0.6, type=float)
    parser.add_argument('--first-aggregate', type=str, default='mean', help='aggregation for hyperedge h_e: max, sum, mean')
    parser.add_argument('--second-aggregate', type=str, default='sum', help='aggregation for node x_i: max, sum, mean')
    parser.add_argument('--use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--type_att_size', default=32, type=int)  # Placeholder
    parser.add_argument('--type_fusion', default='att', type=str)  # Placeholder
    parser.add_argument('--use_node_attn', default='mean', type=str)  # Placeholder
    parser.add_argument('--use_hyperedge_attn', default='mean', type=str)  # Placeholder

    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(AllSet_GPR=False)
    parser.set_defaults(AllSet_LearnMask=False)
    parser.set_defaults(AllSet_PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=True)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    #     Use the line below for notebook
    # args = parser.parse_args([])
    # args, _ = parser.parse_known_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
