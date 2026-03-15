import datetime
import os
import time
import logging
import pickle as pkl

import torch
from deepctr_torch.inputs import get_feature_names

from config.configs import ES_TFI_Config, General_Config
from utils.function_utils import get_param_sum, create_embedding_matrix
from model.ES_TFI_Model import ES_TFI_Model
from utils.function_utils import random_selected_interaction_type



def evolution_search(feature_columns, feature_index,
                     data_train, data_val, param_save_dir, task_labels,
                     embedding_size, runidx, dense_columns=None,
                     mutation=True, device='cpu'):

    if dense_columns is None:
        dense_columns = []
    logging.info('ES-TFI Evolution Search period param:')
    logging.info(ES_TFI_Config['ES_TFI'])

    feature_names = get_feature_names(feature_columns)
    epoch = General_Config['general']['epochs']

    pair_feature_len = int(len(feature_names) * (len(feature_names) - 1) / 2)  # feature pair number 不算自己和自己的
    
    # 为每个任务生成随机交互类型
    num_tasks = len(task_labels)
    selected_interaction_types = [random_selected_interaction_type(pair_feature_len) for _ in range(num_tasks)]

    data_feature_names = get_feature_names(feature_columns + dense_columns)
    train_model_input = {name: data_train[name] for name in data_feature_names}
    val_model_input = {name: data_val[name] for name in data_feature_names}


    train_task_labels = [data_train[label_name].values for label_name in task_labels]
    val_task_labels = [data_val[label_name].values for label_name in task_labels]
    
    shared_embedding = create_embedding_matrix(feature_columns, init_std=0.001)
    logging.info('ES-TFI Evolution Search period start')
    es_tfi_model = ES_TFI_Model(feature_columns=feature_columns, feature_index=feature_index,
                            param_save_dir=param_save_dir, run_idx=runidx,
                            selected_interaction_types=selected_interaction_types,
                            num_tasks=num_tasks,
                            mutation=mutation, shared_embedding=shared_embedding,
                            mutation_probability=ES_TFI_Config['ES_TFI']['mutation_probability'],
                            embedding_size=embedding_size,
                            device=device)
    es_tfi_model.to(device)
    es_tfi_model.before_train()

    start_time = time.time()
    get_param_sum(model=es_tfi_model)

    es_tfi_model.fit(x=train_model_input, task_labels=train_task_labels,
                            val_x=val_model_input,
                            val_task_labels=val_task_labels,
                            batch_size=General_Config['general']['batch_size'],
                            epochs=epoch,
                            )


    end_time = time.time()
    cost_time = int(end_time - start_time)
    logging.info('ES-TFI Evolution Search period end')
    logging.info('ES-TFI Evolution Search period cost:' + str(cost_time))
