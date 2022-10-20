# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, head_size, dropout, name=None, has_bias=True):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = head_size
        self.inner_size = heads_num * head_size

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size, self.inner_size, bias=has_bias) for _ in range(3)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_size, hidden_size, bias=has_bias)
        self.name = name

    def forward(self, key, value, query, mask, cache=None, content_type=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            cache: which is used during inference
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(x.size(0), -1, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        if self.name == "self":
            query, key, value = [shape(l(x)) for l, x in zip(self.linear_layers, (query, key, value))]
            if cache is not None:
                if cache["self_keys"] is not None:
                    key = torch.cat([cache["self_keys"], key], dim=2)
                if cache["self_values"] is not None:
                    value = torch.cat([cache["self_values"], value], dim=2)
                cache["self_keys"] = key
                cache["self_values"] = value
        elif self.name == "context":
            query = shape(self.linear_layers[0](query))
            if cache is not None:
                if content_type is not None:
                    key_name = content_type + "_memory_keys"
                    value_name = content_type + "_memory_values"
                else:
                    key_name = "memory_keys"
                    value_name = "memory_values"
                if cache[key_name] is None:
                    key = shape(self.linear_layers[1](key))
                    value = shape(self.linear_layers[2](value))
                else:
                    key, value = cache[key_name], cache[value_name]
                cache[key_name] = key
                cache[value_name] = value
            else:
                key = shape(self.linear_layers[1](key))
                value = shape(self.linear_layers[2](value))
        else:
            query, key, value = [shape(l(x)) for l, x in zip(self.linear_layers, (query, key, value))]
        query = query / math.sqrt(float(per_head_size))
        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask.size(3) != scores.size(3):
            mask = torch.cat([
                torch.ones(mask.size(0), mask.size(1), mask.size(2), scores.size(3)-mask.size(3), device=mask.device),
                mask], dim=-1)
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = torch.matmul(probs, value).transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length, self.inner_size)
        output = self.final_linear(output)
        return output
