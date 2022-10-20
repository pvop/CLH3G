# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from bert.layers.layer_norm import LayerNorm
from bert.layers.relative_position_embedding import RelativePositionBias





class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size, word_embedding=None, is_seg=False, is_rpe=False, num_head=0, autoregression=False):
        '''
        Args:
            vocab_size: vocabulary size of input for word embedding matrix
            word_embedding: user for shared embedding between encoder and decoder
            is_seg: whether use segment embedding, usually False in NLG model, True in NLU model
            is_rpe: whether use relative embedding
            num_head: relative embedding in T5 is different between heads, so the relative embedding matrix shape is [length, num_head]

        '''
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = args.seq_length
        self.layernorm_position = args.layernorm_position
        self.is_rpe = is_rpe
        has_bias = bool(1 - args.remove_transformer_bias)
        if word_embedding is None:
            self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        else:
            self.word_embedding = word_embedding
        if not is_rpe:
            self.abs_position_embedding = nn.Embedding(args.max_length, args.hidden_size)

        else:
            self.position_embedding = RelativePositionBias(bidirectional=not autoregression, n_heads=num_head)
        if is_seg:
            self.segment_embedding = nn.Embedding(2, args.hidden_size)
        if self.layernorm_position=="post":
            self.layer_norm = LayerNorm(args.hidden_size, has_bias)




    def forward(self, src, seg=None, pos=None):
        """
        index is not None only in Unilm
        index==0: mlm model
        index==1: half for forward lm model and half for backward lm model
        index==2: Seq2Seq Model
        """
        word_emb = self.word_embedding(src)
        if seg is not None:
            seg_emb = self.segment_embedding(seg)
            emb = word_emb + seg_emb
        else:
            emb = word_emb
        if not self.is_rpe:
            if pos is None:
                pos_emb = self.abs_position_embedding(torch.arange(0, word_emb.size(1),
                                                               device=word_emb.device,
                                                               dtype=torch.long).
                                                  unsqueeze(0).
                                                  repeat(word_emb.size(0),1))
            else:
                pos_emb = self.abs_position_embedding(pos)
            emb += pos_emb
        emb = self.dropout(emb)
        if self.layernorm_position=="post":
            emb = self.layer_norm(emb)
        return emb
