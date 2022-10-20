# -*- encoding:utf-8 -*-
import torch.nn as nn
from bert.layers.layer_norm import LayerNorm
from bert.layers.position_ffn import PositionwiseFeedForward
from bert.layers.multi_headed_attn import MultiHeadedAttention
from bert.layers.transformer import TransformerLayer
import torch


class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_position = args.layernorm_position
        has_bias = bool(1 - args.remove_transformer_bias)
        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size, bias=False)

        if self.parameter_sharing:
            self.transformer = TransformerLayer(args)
        else:
            self.transformer = nn.ModuleList([
                TransformerLayer(args) for _ in range(self.layers_num)
            ])
        self.output_dropout = nn.Dropout(args.dropout)
        if self.layernorm_position == "pre":
            self.out_layer_norm = LayerNorm(args.hidden_size, has_bias=has_bias)

    def _init_cache(self):
        self.cache = {}
        for l in range(self.layers_num):
            layer_cache = {
                "self_keys": None,
                "self_values": None
            }
            self.cache["layer_{}".format(l)] = layer_cache

    def map_batch_fn(self, fn):
        for layer in self.cache:
            for key in self.cache[layer]:
                self.cache[layer][key] = fn(self.cache[layer][key], dim=0)

    def forward(self, emb, seg, extra_mask=None, with_cache=False):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            extra_mask: [batch_size x 1 x seq_length x seq_length] or [1 x num_head x seq_length x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)
            
        seq_length = emb.size(1)
        mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)

        mask = mask.float()
        mask = (1.0 - mask).masked_fill(torch.logical_not(mask), float("-inf"))
        if extra_mask is not None:
            mask = mask + extra_mask
        hidden = emb
        for i in range(self.layers_num):
            if self.parameter_sharing:
                if with_cache:
                    hidden = self.transformer(hidden, mask, self.cache["layer_{}".format(i)])
                else:
                    hidden = self.transformer(hidden, mask)
            else:
                if with_cache:
                    hidden = self.transformer[i](hidden, mask, self.cache["layer_{}".format(i)])
                else:
                    hidden = self.transformer[i](hidden, mask)
        if self.layernorm_position == "pre":
            hidden = self.out_layer_norm(hidden)
        return hidden
