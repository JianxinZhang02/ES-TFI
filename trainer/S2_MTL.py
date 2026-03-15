import logging
import math
import os
import sys
import time
import warnings
import pickle as pkl
from config.configs import General_Config, ES_TFI_Config
from layer.inter_AdaTT import inter_AdaTT
from utils.function_utils import log_new, create_uniform_param, create_structure_param, \
    random_selected_interaction_type
import torch
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from deepctr_torch.inputs import get_feature_names
from sklearn.metrics import *

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def model_functioning(
        sparse_feature_columns, sparse_feature_index,
        data_train, data_val, data_test,
        param_save_dir, runidx, seed,
        embedding_size, task_labels, dataset='qb_video', dense_columns=None,
        device='cpu'
):

    if dense_columns is None:
        dense_columns = []
    fixlen_feature_columns = sparse_feature_columns + dense_columns
    feature_names = get_feature_names(fixlen_feature_columns)

    train_model_input = {name: data_train[name] for name in feature_names}
    val_model_input = {name: torch.tensor(data_val[name].values, dtype=torch.float32) for name in feature_names}
    test_model_input = {name: torch.tensor(data_test[name].values, dtype=torch.float32) for name in feature_names}

    pair_feature_len = int(len(sparse_feature_columns) * (len(sparse_feature_columns) - 1) / 2)

    # Load interaction types for all tasks dynamically
    num_tasks = len(task_labels)
    selected_interaction_types = []
    
    for task_idx in range(num_tasks):
        interaction_type_path = os.path.join(
            param_save_dir, 
            f'interaction_type_task{task_idx}-embedding_size-{embedding_size}.pkl'
        )
        if os.path.exists(interaction_type_path):
            interaction_type = pkl.load(open(interaction_type_path, 'rb'))
            logging.info(f"[OK] Loaded interaction type for task {task_idx} from {interaction_type_path}")
        else:
            logging.warning(f"[WARN] Interaction type file not found for task {task_idx} at {interaction_type_path}. Using random interaction type.")
            interaction_type = random_selected_interaction_type(pair_feature_len)
        selected_interaction_types.append(interaction_type)
        selected_interaction_types.append(interaction_type)

    save_dir = os.path.join('param', dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'run_{runidx}.pth')
    patience = General_Config['general']['early_stopping_epoch']
    early_stopping = EarlyStopping(monitor='val_auc', patience=patience, mode='max')
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_path),
        monitor='val_auc',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    )

    model = inter_AdaTT(
        dnn_features=fixlen_feature_columns, 
        expert_out_dims=[[256, 128], [256, 128]], 
        num_tasks=num_tasks,
        num_task_experts=2, 
        num_shared_experts=2,
        task_mlp=[64],
        self_exp_res_connect=True,
        input_dim=General_Config['general']['qb_video_embedding_size'],
        sparse_feature_columns=sparse_feature_columns,
        sparse_feature_index=sparse_feature_index,
        selected_interaction_types=selected_interaction_types,
        seed=seed,
        device=device
    )

    logging.info("Model Functioning period start")
    model.to(device)

    loss_list = ["binary_crossentropy" for _ in range(num_tasks)]
    model.compile("adam", loss=loss_list, metrics=['binary_crossentropy', 'auc'])
    start_time = time.time()
    history = model.fit(
        train_model_input, data_train[task_labels].values,
        validation_data=(val_model_input, data_val[task_labels].values),
        batch_size=General_Config['general']['phase_2_batch'],
        epochs=General_Config['general']['epochs'],
        verbose=1,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    model.load_state_dict(torch.load(save_path, weights_only=True))

    pred_ans = model.predict(test_model_input, 2048)

    results = {}
    for i, target_name in enumerate(task_labels):
        logloss = round(log_loss(data_test[target_name].values, pred_ans[:, i]), 4)
        auc = round(roc_auc_score(data_test[target_name].values, pred_ans[:, i]), 4)
        results[target_name] = {'LogLoss': logloss, 'AUC': auc}
        logging.info(f"{target_name} - LogLoss: {logloss}, AUC: {auc}")
    print(results)

    end_time = time.time()
    cost_time = int(end_time - start_time)
    logging.info("Model Functioning period end")
    logging.info('Model Functioning period cost:' + str(cost_time))

    return results
