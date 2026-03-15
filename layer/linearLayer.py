import torch
import torch.nn as nn
from utils.function_utils import create_embedding_matrix



class NormalizedWeightedLinearLayer(nn.Module):  # 归一化加权线性层，处理 稀疏特征（Sparse Features） 的线性变换
    def __init__(self, feature_columns, feature_index, alpha=None, use_alpha=True, tag=False, second_tag=False,
                 embedding_dict=None,
                 alpha_activation='tanh', device='cpu'):
        super(NormalizedWeightedLinearLayer, self).__init__()

        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.device = device
        self.use_alpha = use_alpha
        self.tag = tag
        if embedding_dict is None:
            self.embedding_dict = create_embedding_matrix(feature_columns, init_std=0.001, sparse=False)
        else:
            self.embedding_dict = embedding_dict

        if alpha == None:
            self.alpha = self.create_structure_param(len(self.feature_columns), init_mean=0.5, init_radius=0.001)
        else:
            self.alpha = alpha
        self.activate = nn.Tanh() if alpha_activation == 'tanh' else nn.Identity()

    def create_structure_param(self, length, init_mean, init_radius):
        structure_param = nn.Parameter(
            torch.empty(length).uniform_(
                init_mean - init_radius,
                init_mean + init_radius))
        structure_param.requires_grad = True
        return structure_param

    def create_uniform_param(self, size, init_range=(0.0, 1.0)):
        param = nn.Parameter(torch.empty(size, device=self.device).uniform_(*init_range))
        param.requires_grad = True
        return param

    def forward(self, X):
        embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()
        ) 
            for feat in self.feature_columns
        ]

        if self.use_alpha:
            xxx = torch.cat(embedding_list, dim=1).squeeze(
                -1) 
            yyy = (self.activate(self.alpha)) 
            zzz = yyy.unsqueeze(0).unsqueeze(-1)
            mul_result = torch.mul(xxx, zzz)
            if self.tag == True:
                return mul_result
            linear_logit_orignal = torch.sum(mul_result, dim=(1, 2))
            linear_logit = linear_logit_orignal.squeeze().unsqueeze(-1)
        else:
            linear_logit = torch.flatten(torch.cat(embedding_list, dim=1), start_dim=1)

        return linear_logit
