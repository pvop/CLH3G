# -*- encoding:utf-8 -*-
import torch
from bert.layers.embeddings import BertEmbedding
from bert.encoders.bert_encoder import BertEncoder
from bert.models.pegasus_model_stackfuse import PegasusModelStackFuse
from bert.targets.pegasus_target_decoder_multi_input import PegasusTargetDecoderMultiInput
from bert.decoders.transformer_decoder_multi_input import TransformerDecoderMultiInput
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
    decoder = TransformerDecoderMultiInput(args)
    target = PegasusTargetDecoderMultiInput(args, len(args.vocab))
    model = PegasusModelStackFuse(args, src_embedding, tgt_embedding, encoder, decoder, target, args.vocab.get(START_TOKEN),
                    vocab_size=len(args.vocab), end_id=args.vocab.get(END_TOKEN))
    return model

