import time

import pandas as pd
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import *
from torch.utils.data import DataLoader, TensorDataset
from utils.function_utils import slice_arrays
from config.configs import General_Config, ES_TFI_Config


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, x, y, batch_size=256, task_idx=0):

        pred_ans = self.predict_single_task(x, batch_size, task_idx)
        
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        
        return eval_result
    
    def evaluate_all_tasks(self, x, task_labels_list, batch_size=256):

        all_task_preds = self.predict_all_tasks(x, batch_size)
        
        task_eval_results = []
        for task_idx, (y_true, y_pred) in enumerate(zip(task_labels_list, all_task_preds)):
            eval_result = {}
            for name, metric_fun in self.metrics.items():
                eval_result[name] = metric_fun(y_true, y_pred)
            task_eval_results.append(eval_result)
        
        return task_eval_results
    
    def predict_all_tasks(self, x, batch_size=256):

        self.eval()
        
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        
        tensor_data = TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1))
        )
        
        test_loader = DataLoader(
            dataset=tensor_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4
        )
        
        task_predictions = None
        
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                outputs = self(x)
                
                if isinstance(outputs, tuple):
                    num_tasks = len(outputs)
                    if task_predictions is None:
                        task_predictions = [[] for _ in range(num_tasks)]
                    
                    for task_idx in range(num_tasks):
                        task_logit = outputs[task_idx]
                        task_prob = torch.sigmoid(task_logit).cpu().numpy()
                        task_predictions[task_idx].append(task_prob)
                else:
                    if task_predictions is None:
                        task_predictions = [[]]
                    task_prob = torch.sigmoid(outputs).cpu().numpy()
                    task_predictions[0].append(task_prob)
        
        return [np.concatenate(preds).astype("float64") for preds in task_predictions]
    
    def predict_single_task(self, x, batch_size=256, task_idx=0):

        self.eval()
        
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        
        tensor_data = TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1))
        )
        
        test_loader = DataLoader(
            dataset=tensor_data,
            shuffle=False,
            batch_size=batch_size
        )
        
        predictions = []
        
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                outputs = self(x) 
                if isinstance(outputs, tuple):
                    if task_idx >= len(outputs):
                        raise ValueError(f"task_idx {task_idx} 超出模型任务数量 {len(outputs)}")
                    task_logit = outputs[task_idx]
                else:
                    if task_idx != 0:
                        raise ValueError(f"单任务模型不支持 task_idx={task_idx}")
                    task_logit = outputs
                
                task_prob = torch.sigmoid(task_logit).cpu().numpy()
                predictions.append(task_prob)
        
        return np.concatenate(predictions).astype("float64")

    def predict(self, x, batch_size=256):
        self.eval()

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1))
        )

        test_loader = DataLoader(
            dataset=tensor_data,
            shuffle=False,
            batch_size=batch_size
        )

        pred_y, pred_z = [], []

        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_logit, z_logit = self(x)

                y_prob = torch.sigmoid(y_logit).cpu().numpy()
                z_prob = torch.sigmoid(z_logit).cpu().numpy()

                pred_y.append(y_prob)
                pred_z.append(z_prob)

        return np.concatenate(pred_y).astype("float64"), np.concatenate(pred_z).astype("float64")

