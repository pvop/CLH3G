# -*- encoding:utf-8 -*-
import torch
from bert.layers.embeddings import BertEmbedding, BertEmbeddingMixGrained
from bert.encoders.bert_encoder import BertEncoder
from bert.models.contra_model import ContraModel
from bert.targets.contra_target import ContraTarget

def build_model(args):
    """
    Build universial encoder representations models.
    Only BERT-like models are retained in this project.
    The combinations of different embedding, encoder,
    and target layers yield pretrained models of different
    properties.
    We could select suitable one for downstream tasks.
    """


    src_embedding = BertEmbedding(args, len(args.vocab), is_rpe=False,is_seg=True, num_head=args.heads_num, autoregression=False)
    encoder = BertEncoder(args)
    target = ContraTarget(args)
    model = ContraModel(src_embedding, encoder, target)
    return model

