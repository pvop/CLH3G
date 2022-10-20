# -*- encoding:utf-8 -*-
import torch.nn as nn
import torch
from bert.layers.multi_headed_attn import MultiHeadedAttention
from bert.layers.layer_norm import LayerNorm

class KeywordAttn(nn.Module):
    def __init__(self, args):
        super(KeywordAttn, self).__init__()
        has_bias = bool(1 - args.remove_transformer_bias)
        self.self_attn = MultiHeadedAttention(args.hidden_size, args.heads_num, args.head_size, args.dropout, has_bias=has_bias)
        self.layer_norm = LayerNorm(args.hidden_size, has_bias=has_bias)
    def forward(self, src_hidden, tgt_hidden, src, tgt):
        src_seq_length = src_hidden.size(1)
        tgt_seq_length = tgt_hidden.size(1)

        mask_src = (src > 0). \
            unsqueeze(2). \
            repeat(1, 1, tgt_seq_length). \
            unsqueeze(1)
        mask_src = mask_src.float()
        mask_src = (1.0 - mask_src) * -10000.0

        mask_tgt = (tgt > 0). \
            unsqueeze(1). \
            repeat(1, src_seq_length, 1). \
            unsqueeze(1)
        mask_tgt = mask_tgt.float()
        mask_tgt = (1.0 - mask_tgt) * -10000.0
        mask = mask_tgt + mask_src

        result = self.self_attn(tgt_hidden, tgt_hidden, src_hidden, mask)
        result = self.layer_norm(src_hidden + result)
        return result