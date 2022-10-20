# -*- encoding:utf-8 -*-
import torch.nn as nn
from bert.utils import str2act


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """
    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = str2act[hidden_act]
        
    def forward(self, x):
        inter = self.act(self.linear_1(x))
        output = self.linear_2(inter)
        return output

class GatedFeedFoward(nn.Module):
    """ Feed Forward Layer with Gated Linear Unit.
    From: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(GatedFeedFoward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_gate = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = str2act[hidden_act]
    def forward(self, x):
        inter_gate = self.act(self.linear_1(x))
        inter_linear = self.linear_gate(x)
        output = self.linear_2(inter_gate * inter_linear)
        return output
