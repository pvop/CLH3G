# -*- encoding:utf-8 -*-
import torch.nn as nn
from bert.layers.layer_norm import LayerNorm
from bert.layers.position_ffn import PositionwiseFeedForward, GatedFeedFoward
from bert.layers.multi_headed_attn import MultiHeadedAttention
import torch


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        # Multi-headed self-attention.
        self.layernorm_position = args.layernorm_position
        has_bias = bool(1 - args.remove_transformer_bias)

        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.head_size, args.dropout, name="self", has_bias=has_bias
        )
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_1 = LayerNorm(args.hidden_size, has_bias=has_bias)

        # Feed forward layer.
        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedFoward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias=has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias=has_bias
            )
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size, has_bias=has_bias)

    def forward(self, hidden, mask, cache=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        if self.layernorm_position == "post":
            inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask, cache=cache))
            inter = self.layer_norm_1(inter + hidden)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        elif self.layernorm_position == "pre":
            inter = self.layer_norm_1(hidden)
            inter = self.dropout_1(self.self_attn(inter, inter, inter, mask, cache=cache))
            inter = inter + hidden
            output = self.layer_norm_2(inter)
            output = self.dropout_2(self.feed_forward(output))
            output = output + inter
        else:
            raise KeyError("Input layernorm_position is neither \"post\" nor \"pre\".")
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()
        self.layernorm_position = args.layernorm_position
        has_bias = bool(1 - args.remove_transformer_bias)
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.head_size, args.dropout, name="self", has_bias=has_bias
        )
        self.context_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.head_size, args.dropout, name="context", has_bias=has_bias
        )
        self.layer_norm_1 = LayerNorm(args.hidden_size, has_bias=has_bias)
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size, has_bias=has_bias)
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_3 = LayerNorm(args.hidden_size, has_bias=has_bias)
        self.dropout_3 = nn.Dropout(args.dropout)
        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedFoward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias=has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias=has_bias
            )
    def forward(self, hidden, encoder_hidden, mask_decoder, mask_encoder, cache=None, st=False):
        """
                Args:
                    emb: [batch_size x seq_length x emb_size]
                    hidden: [batch_size x seq_length x emb_size]
                    mask_encoder: [batch_size x 1 x tgt_length x src_length]
                    mask_decoder: [batch_size x 1 x tgt_length x tgt_length]
                Returns:
                    output: [batch_size x seq_length x hidden_size]
        """
        if self.layernorm_position=="post":
            query = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask_decoder, cache=cache))
            query_norm = self.layer_norm_1(query + hidden)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, cache=cache))
            mid_norm = self.layer_norm_2(mid + query_norm)
            output = self.dropout_3(self.feed_forward(mid_norm))
            output = self.layer_norm_3(output + mid_norm)
        else:
            hidden_norm = self.layer_norm_1(hidden)
            query = self.dropout_1(self.self_attn(hidden_norm, hidden_norm, hidden_norm, mask_decoder, cache=cache))
            query = query + hidden
            query_norm = self.layer_norm_2(query)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, cache=cache))
            mid = mid + query
            mid_norm = self.layer_norm_3(mid)
            output = self.dropout_3(self.feed_forward(mid_norm))
            output = output + mid
        return output

class TransformerDecoderMultiInputLayer(nn.Module):
    def __init__(self, args):
        super(TransformerDecoderMultiInputLayer, self).__init__()
        self.layernorm_position = args.layernorm_position
        has_bias = bool(1 - args.remove_transformer_bias)
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.head_size, args.dropout, name="self", has_bias=has_bias
        )
        self.art_context_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.head_size, args.dropout, name="context", has_bias=has_bias
        )
        self.title_context_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.head_size, args.dropout, name="context", has_bias=has_bias
        )
        self.layer_norm_1 = LayerNorm(args.hidden_size, has_bias=has_bias)
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size, has_bias=has_bias)
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_3 = LayerNorm(args.hidden_size, has_bias=has_bias)
        self.dropout_3 = nn.Dropout(args.dropout)
        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedFoward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias=has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias=has_bias
            )
    def forward(self, hidden, art_encoder_outputs, title_encoder_outputs, mask_decoder, art_mask, title_mask, cache=None, st=False):
        """
                Args:
                    emb: [batch_size x seq_length x emb_size]
                    hidden: [batch_size x seq_length x emb_size]
                    mask_encoder: [batch_size x 1 x tgt_length x src_length]
                    mask_decoder: [batch_size x 1 x tgt_length x tgt_length]
                Returns:
                    output: [batch_size x seq_length x hidden_size]
        """
        if self.layernorm_position=="post":
            query = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask_decoder, cache=cache))
            query_norm = self.layer_norm_1(query + hidden)

            title_mid = self.dropout_2(self.title_context_attn(title_encoder_outputs, title_encoder_outputs, query_norm, title_mask, cache=cache, content_type="title"))

            art_mid = self.dropout_3(
                self.art_context_attn(art_encoder_outputs, art_encoder_outputs, query_norm, art_mask, cache=cache,
                                  content_type="art"))

            art_mid_norm = self.layer_norm_2(art_mid + title_mid + query_norm)
            output = self.dropout_3(self.feed_forward(art_mid_norm))
            output = self.layer_norm_3(output + art_mid_norm)
        return output