import os, sys
import matplotlib.pyplot as plt

import numpy as np

import torch
from sklearn.metrics import (average_precision_score,
                             roc_auc_score,
                             f1_score,
                             normalized_mutual_info_score,
                             adjusted_rand_score,
                             accuracy_score)
class NodeClsEvaluator:

    def __init__(self):
        return

    def eval(self, y_true, y_pred):
        acc_list = []
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

        is_labeled = (~np.isnan(y_true)) & (~np.isinf(y_true)) # no nan and inf
        correct = (y_true[is_labeled] == y_pred[is_labeled])
        acc_list.append(float(np.sum(correct))/len(correct))
        """
        micro = f1_score(y_true, y_pred, average='micro')
        macro = f1_score(y_true, y_pred, average='macro')
        print("micro:",micro)
        print("macro:",macro)
        """

        return {'acc': sum(correct) / sum(is_labeled)}

class NodeRegEvaluator:

    def __init__(self):
        return

    def eval(self, y_true, y_pred):
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()
        d = y_true - y_pred
        return {
            'mse': torch.mean(torch.square(d)).item(),
            'mae': torch.mean(torch.abs(d)).item(),
            'mape': torch.mean(torch.abs(d) / torch.abs(y_true)).item(),
        }

""" Adapted from https://github.com/snap-stanford/ogb/ """
class Logger:

    def __init__(self, runs, log_path=None):
        self.log_path = log_path
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, train_acc, valid_acc, test_acc, micro, macro, auc):
        result = [train_acc, valid_acc, test_acc,micro, macro, auc]
        assert len(result) == 6
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            max_train = result[:, 0].max().item()
            max_test = result[:, 2].max().item()

            argmax = result[:, 1].argmax().item()
            train = result[argmax, 0].item()
            valid = result[argmax, 1].item()
            test = result[argmax, 2].item()

            micro = result[argmax, 3].item()
            macro = result[argmax, 4].item()
            auc = result[argmax, 5].item()
            return {'max_train': max_train, 'max_test': max_test,
                'train': train, 'valid': valid, 'test': test, 'micro': micro, 'macro': macro,'auc': auc}
        else:
            keys = ['max_train', 'max_test', 'train', 'valid', 'test','micro', 'macro', 'auc']

            best_results = []
            for r in range(len(self.results)):
                best_results.append([self.get_statistics(r)[k] for k in keys])

            ret_dict = {}
            best_result = torch.tensor(best_results)
            for i, k in enumerate(keys):
                ret_dict[k+'_mean'] = best_result[:, i].mean().item()
                ret_dict[k+'_std'] = best_result[:, i].std().item()

            return ret_dict

    def print_statistics(self, run=None):
        if run is not None:
            result = self.get_statistics(run)
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result['max_train']:.2f}")
            print(f"Highest Valid: {result['valid']:.2f}")
            print(f"  Final Train: {result['train']:.2f}")
            print(f"   Final Test: {result['test']:.2f}")
        else:
            result = self.get_statistics()
            print(f"All runs:")
            print(f"Highest Train: {result['max_train_mean']:.2f} ± {result['max_train_std']:.2f}")
            print(f"Highest Valid: {result['valid_mean']:.2f} ± {result['valid_std']:.2f}")
            print(f"  Final Train: {result['train_mean']:.2f} ± {result['train_std']:.2f}")
            print(f"   Final Test: {result['test_mean']:.2f} ± {result['test_std']:.2f}")
            print(f"   Final micro: {result['micro_mean']:.2f} ± {result['micro_std']:.2f}")
            print(f"   Final macro: {result['macro_mean']:.2f} ± {result['macro_std']:.2f}")
            print(f"   Final auc: {result['auc_mean']:.2f} ± {result['auc_std']:.2f}")

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results).mean(0)
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])

""" Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
""" randomly splits label into train/valid/test splits """
def rand_train_test_idx(label, train_prop, valid_prop, balance=False):
    if not balance:
        n = label.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.randperm(n)

        train_idx = perm[:train_num]
        valid_idx = perm[train_num:train_num + valid_num]
        test_idx = perm[train_num + valid_num:]

        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }

    else:
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([ind[:percls_trn] for ind in indices], dim=0)
        rest_index = torch.cat([ind[percls_trn:] for ind in indices], dim=0)
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]

        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }

    return split_idx

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import HeteroData
from torch_geometric.datasets import IMDB, AMiner
from torch_geometric.transforms import AddMetaPaths
from torch_geometric.utils import to_undirected, add_self_loops
from itertools import permutations  

from torch_sparse import SparseTensor


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None,
                     test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size,
        val_size, test_size)

    # print('number of training: {}'.format(len(train_indices)))
    # print('number of validation: {}'.format(len(val_indices)))
    # print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['valid'] = val_mask
    mask['test'] = test_mask
    return mask
def clique_expansion(hyperedge_index):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'ES_tmp_files/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))