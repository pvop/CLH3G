# -*- encoding:utf-8 -*-
import torch
from bert.layers.embeddings import BertEmbedding
from bert.encoders.bert_encoder import BertEncoder
from bert.models.pegasus_model_title_vector_contra import PegasusModelTitleVectorContra
from bert.targets.pegasus_target_title_vector import PegasusTargetTitleVector
from bert.targets.pegasus_target import PegasusTarget
from bert.targets.contra_target import ContraTarget
from bert.decoders.transformer_decoder import TransformerDecoder
from bert.utils.constants import START_TOKEN, END_TOKEN

def build_model(args):
    """
    Build universial encoder representations models.
    Only BERT-like models are retained in this project.
    The combinations of different embedding, encoder,
    and target layers yield pretrained models of different
    properties.
    We could select suitable one for downstream tasks.
    """


    src_embedding = BertEmbedding(args, len(args.vocab), is_rpe=False, is_seg=True)
    tgt_embedding = BertEmbedding(args, len(args.vocab), word_embedding=src_embedding.word_embedding, is_seg=True)
    encoder = BertEncoder(args)
    decoder = TransformerDecoder(args)
    if args.pointer_title_vector:
        target = PegasusTargetTitleVector(args, len(args.vocab))
    else:
        target = PegasusTarget(args, len(args.vocab))
    contra_target = ContraTarget(args)
    model = PegasusModelTitleVectorContra(args, src_embedding, tgt_embedding, encoder, decoder, target, contra_target, args.vocab.get(START_TOKEN),
                    vocab_size=len(args.vocab), end_id=args.vocab.get(END_TOKEN))
    return model

