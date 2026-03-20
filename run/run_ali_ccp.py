

import numpy as np
import pickle as pkl
import random
import time
import logging
import torch
import pandas as pd
from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names, build_input_features
from utils.function_utils import log, set_random_seed
from trainer.S1_Evo import evolution_search
from trainer.S2_MTL import model_functioning
from config.configs import General_Config, ES_TFI_Config
import os

feat_names_sparse = [
    '101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '210',
    '216', '508', '509', '702', '853', '301', '109_14', '110_14', '127_14', '150_14',
]  # 23个特征

feat_names_dense = [
    'D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853'
]  # 8个稠密特征


def train(params):
    target = params.label
    param_save_dir = os.path.join('../param/ali_ccp',
                                  time.strftime("%Y-%m-%d-%H-%M", time.localtime()))
    if not os.path.exists(param_save_dir):
        param_save_dir_fis_type = os.path.join(param_save_dir, "evolution", "operation_type")
        param_save_dir_alphabeta = os.path.join(param_save_dir, "evolution", "alpha_beta")
        os.makedirs(param_save_dir_fis_type)
        os.makedirs(param_save_dir_alphabeta)

    log(label=params.label, dataset=params.dataset, model=params.model)
    logging.info('-' * 50)
    logging.info(str(time.asctime(time.localtime(time.time()))))
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        logging.info('cuda ready...')
        device = 'cuda:' + str(params.gpu)
    logging.info(params)
    logging.info(General_Config)

    data_train = pd.read_pickle('../data/Ali-CCP/train.pkl')
    data_val = pd.read_pickle('../data/Ali-CCP/val.pkl')
    data_test = pd.read_pickle('../data/Ali-CCP/test.pkl')

    feat_size = pkl.load(open('../Ali-CCP/random_data/feature_size.pkl', 'rb'))
    sparse_feature_columns = [
        SparseFeat(feat_name, feat_size[feat_name], General_Config['general']['ali_ccp_embedding_size'])
        for feat_name in feat_names_sparse]
    dense_feature_columns = [
        DenseFeat(feat_name)
        for feat_name in feat_names_dense
    ]
    sparse_feature_index = build_input_features(sparse_feature_columns)

    mutation = bool(params.mutation)
    seed = random.randint(0, 1000)
    set_random_seed(seed)
    logging.info(f"Seed {seed}")

    evolution_search(
        feature_columns=sparse_feature_columns, feature_index=sparse_feature_index,
        data_train=data_train, data_val=data_val, param_save_dir=param_save_dir,
        mutation=mutation, task_labels=target,
        runidx=0,
        embedding_size=General_Config['general']['ali_ccp_embedding_size'],
        device=device
    )

    model_functioning(
        sparse_feature_columns=sparse_feature_columns,
        sparse_feature_index=sparse_feature_index,
        data_train=data_train, runidx=0,
        data_val=data_val, data_test=data_test,
        param_save_dir=param_save_dir, seed=seed,
        embedding_size=General_Config['general']['ali_ccp_embedding_size'],
        task_labels=target,
        dataset='ali_ccp',
        dense_columns=dense_feature_columns,
        device=device
    )
