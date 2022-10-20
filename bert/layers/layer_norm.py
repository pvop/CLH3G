# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    '''
        has_bias==False: Root Mean Square Layer Normalization: https://arxiv.org/pdf/1910.07467 for T5
    '''
    def __init__(self, hidden_size, has_bias=True, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.has_bias = has_bias
        if has_bias:
            self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        if self.has_bias:
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta
        else:
            rms = torch.sqrt(torch.mean(torch.square(x),dim=-1, keepdim=True))
            return self.gamma * x / (rms+self.eps)
