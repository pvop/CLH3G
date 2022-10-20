# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ContraModel(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """

    def __init__(self, embedding, encoder, target):
        super(ContraModel, self).__init__()
        self.src_embedding = embedding
        self.encoder = encoder
        self.target = target

    def forward(self, src):
        contra_num = src.size(1)
        src = src.view(-1, src.size(-1)).contiguous()
        seg = torch.ones_like(src)
        src_emb = self.src_embedding(src, seg=seg)
        encoder_output = self.encoder(src_emb, src)
        encoder_output = encoder_output[:, 0]
        encoder_output = encoder_output.view(-1, contra_num, encoder_output.size(-1)).contiguous()
        loss_info = self.target(encoder_output)
        return loss_info
    def forword_inference(self, src):
        contra_num = src.size(1)
        src = src.view(-1, src.size(-1)).contiguous()
        seg = torch.ones_like(src)
        src_emb = self.src_embedding(src, seg=seg)
        encoder_output = self.encoder(src_emb, src)
        encoder_output = encoder_output[:, 0]
        encoder_output = encoder_output.view(-1, contra_num, encoder_output.size(-1)).contiguous()
        results = self.target.inference_forward(encoder_output)
        return results