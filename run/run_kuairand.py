import random
import time
import logging
import numpy as np
import torch
import pandas as pd
from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names, build_input_features
from utils.function_utils import log, set_random_seed
from trainer.S1_Evo import evolution_search
from trainer.S2_MTL import model_functioning
from config.configs import General_Config, ES_TFI_Config
import os
import itertools

sparse_features = [
    'user_id', 'video_id', 'is_rand', 'tab', 'user_active_degree',
    'is_lowactive_period', 'is_live_streamer', 'is_video_author',
    'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range',
    'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
    'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6',
    'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10',
    'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
    'onehot_feat15', 'onehot_feat16', 'onehot_feat17', 'author_id',
    'video_type', 'music_id', 'music_type'
]
dense_fields = []

def train(params):
    param_save_dir = os.path.join('../param/kuairand',
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

    data_train = pd.read_csv('../kuairand/processed/kuairand_train.csv')
    data_val = pd.read_csv('../kuairand/processed/kuairand_val.csv')
    data_test = pd.read_csv('../kuairand/processed/kuairand_test.csv')

    data = pd.concat([data_train, data_val, data_test], axis=0)

    # target = ['is_click', 'is_like']  # 2任务示例long_view  long_view  is_click
    target = params.label
    
    sparse_feature_columns = [
        SparseFeat(feat, data[feat].max() + 1, General_Config['general']['kuairand_embedding_size'])
        for feat in sparse_features
    ]

    dense_feature_columns = [
        DenseFeat(feat, 1)
        for feat in dense_fields
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
        embedding_size=General_Config['general']['kuairand_embedding_size'],
        device=device
    )

    model_functioning(
        sparse_feature_columns=sparse_feature_columns,
        sparse_feature_index=sparse_feature_index,
        data_train=data_train,
        runidx=0,
        data_val=data_val,
        data_test=data_test,
        param_save_dir=param_save_dir, seed=seed,
        embedding_size=General_Config['general']['kuairand_embedding_size'],
        dense_columns=dense_feature_columns,
        task_labels=target,
        dataset='kuairand',
        device=device
    )
