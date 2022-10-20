# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from bert.layers.layer_norm import LayerNorm
from bert.utils.act_fun import gelu


class ClsTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args):
        super(ClsTarget, self).__init__()
        self.cls_num = args.cls_num
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size

        self.mlm_linear = nn.Linear(args.hidden_size, self.cls_num)

        self.softmax = nn.LogSoftmax(dim=-1)

        self.criterion = nn.NLLLoss()

    def mlm(self, memory_bank, tgt_mlm):
        memory_bank = memory_bank[:, 0]
        # Masked language modeling (MLM) with full softmax prediction.
        output_mlm = self.mlm_linear(memory_bank)
        output_mlm = self.softmax(output_mlm)

        one_hot = torch.zeros(output_mlm.size(0), self.cls_num). \
            to(torch.device(output_mlm.device)). \
            scatter_(1, tgt_mlm.contiguous().view(-1, 1), 1.0)
        numerator = -torch.sum(output_mlm * one_hot, 1)
        loss_mlm = torch.sum(numerator) / output_mlm.size(0)
        return loss_mlm, 0, 0

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]
        Returns:
            loss: Masked language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language model (MLM).
        loss, correct, denominator = self.mlm(memory_bank, tgt)

        return loss, correct, denominator

    def inference_forward(self, memory_bank):
        memory_bank = memory_bank[:, 0]
        output_mlm = self.mlm_linear(memory_bank)
        return output_mlm
