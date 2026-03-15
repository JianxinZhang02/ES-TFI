import copy
import logging
import math
import os
import random
import time
from collections import OrderedDict
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from deepctr_torch.inputs import SparseFeat
from torch.optim.optimizer import Optimizer

from data.featureDefiniton import DenseFeat, DenseBucketFeat


def pairwise_to_feature_embedding_full(inter_emb, pair_indexes, num_features):

    feat_i, feat_j = pair_indexes  # [num_pairs]
    B, num_pairs, D = inter_emb.shape
    device = inter_emb.device

    agg = torch.zeros(B, num_features, D, device=device)

    agg.index_add_(1, feat_i, inter_emb)
    agg.index_add_(1, feat_j, inter_emb)

    count = (num_features - 1)

    feature_inter_emb = agg / count

    return feature_inter_emb


class LoRAAdapter(nn.Module):
    def __init__(self, embed_dim, rank=8):
        super().__init__()
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up = nn.Linear(rank, embed_dim, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(rank))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        delta = self.up(self.down(x))
        return x + delta


def build_input_features(feature_columns):
    features = OrderedDict()
    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, DenseBucketFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def create_embedding_matrix(feature_columns, init_std=0.001, linear=False, sparse=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    dense_bucket_feature_columns = list(
        filter(lambda x: isinstance(x, DenseBucketFeat), feature_columns)) if len(feature_columns) else []

    embedding_feature_list = sparse_feature_columns + dense_bucket_feature_columns

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in embedding_feature_list}
    )

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


def generate_pair_index(n, order=2, selected_pairs=None):
    """Return enumeration of feature combination pair index.

    :param n: number of valid features, usually equals to `input_dim4lookup`
    :type n: int
    :param order: order of interaction. defaults to 2
    :type order: int
    :param selected_pairs: specifying selected pair of index
    :type selected_pairs: sequence of tuples, optional
    :return: a list of tuple, each containing feature index
    :rtype: list of tuple

    :Example:

    >>> generate_pair_index(5, 2)
    >>> [(0, 0, 0, 0, 1, 1, 1, 2, 2, 3),
         (1, 2, 3, 4, 2, 3, 4, 3, 4, 4)]
    >>> generate_pair_index(5, 3)
    >>> [(0, 0, 0, 0, 0, 0, 1, 1, 1, 2),
         (1, 1, 1, 2, 2, 3, 2, 2, 3, 3),
         (2, 3, 4, 3, 4, 4, 3, 4, 4, 4)]
    >>> generate_pair_index(5, 2, [(0,1),(1,3),(2,3)])
    >>> [(0, 1, 2), (1, 3, 3)]
    """
    if n < 2:
        raise ValueError("undefined. please ensure n >= 2")
    pairs = list(combinations(range(n), order))
    if selected_pairs is not None and len(selected_pairs) > 0:
        valid_pairs = set(selected_pairs)
        pairs = list(filter(lambda x: x in valid_pairs, pairs))
        print("Using following selected feature pairs \n{}".format(pairs))
        if len(pairs) != len(selected_pairs):
            print("Pair number {} != specified pair number {}".format(len(pairs), len(selected_pairs)))
    return list(zip(*pairs))


def log_new(dataset, model):
    result_save_dir = os.path.join('./result', dataset, time.strftime("%Y-%m-%d", time.localtime()))
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)
    filename = str(model) + '_result_' + str(
        time.strftime("%H-%M-%S", time.localtime())) + '.log'
    logging.basicConfig(filename=os.path.join(result_save_dir, filename), level=logging.INFO, filemode='w')


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def combined_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def slice_arrays(arrays, start=None, stop=None):
    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def get_param_sum(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    logging.info("The total number of parameters:" + str(k))


def log(dataset, model, label):
    result_save_dir = os.path.join('../result', dataset, time.strftime("%Y-%m-%d", time.localtime()))
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)
    filename = str(model) + '_' + label[0] + '_' + label[1] + '_result_' + str(
        time.strftime("%H-%M-%S", time.localtime())) + '.log'
    logging.basicConfig(filename=os.path.join(result_save_dir, filename), level=logging.INFO, filemode='w')


def random_selected_interaction_type(pair_feature_len):
    selected_interaction_type = np.random.rand(pair_feature_len)
    for i in range(pair_feature_len):
        if selected_interaction_type[i] < 0.25:
            selected_interaction_type[i] = 0
        elif selected_interaction_type[i] < 0.5:
            selected_interaction_type[i] = 1
        elif selected_interaction_type[i] < 0.75:
            selected_interaction_type[i] = 2
        else:
            selected_interaction_type[i] = 3
    return torch.tensor(np.array(selected_interaction_type, dtype=int))



class EarlyStopping:
    def __init__(self, patience=2, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_model_state = None
        self.best_custom_state: dict = {}
        self.counter = 0


    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self._save_best_model(model, current_score)
            return False
        elif current_score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = current_score
            self._save_best_model(model, current_score)
            self.counter = 0
        return False

    def _save_best_model(self, model, score):
        self.best_score = score
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.best_custom_state = copy.deepcopy(self._extract_custom_attributes(model))

    def _extract_custom_attributes(self, model):
        if hasattr(model, 'interaction_operations') and hasattr(model, 'num_tasks'):

            selected_types = []
            for task_idx in range(model.num_tasks):
                selected_types.append(
                    model.interaction_operations[task_idx].selected_interaction_type.clone()
                )
            return {
                "num_tasks": model.num_tasks,
                "selected_interaction_types": selected_types
            }
        else:
            return {}

    def restore_model(self, model):
        model.load_state_dict(self.best_model_state)
        if self.best_custom_state:
            if "selected_interaction_types" in self.best_custom_state:
                for task_idx in range(self.best_custom_state["num_tasks"]):
                    model.interaction_operations[task_idx].selected_interaction_type = \
                        self.best_custom_state["selected_interaction_types"][task_idx]
                    model.interaction_operations[task_idx].mask_weight = \
                        model.interaction_operations[task_idx].generate_mask_weight()
        print("Model and custom attributes restored!")



def set_random_seed(seed):
    print("* random_seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_structure_param(length, init_mean, init_radius):
    structure_param = nn.Parameter(
        torch.empty(length).uniform_(
            init_mean - init_radius,
            init_mean + init_radius))
    structure_param.requires_grad = True
    return structure_param


def create_uniform_param(size, init_range=(0.0, 1.0)):
    param = nn.Parameter(torch.empty(size).uniform_(*init_range))
    param.requires_grad = True
    return param
