import torch
import torch.nn as nn
from bert.layers.transformer import TransformerDecoderMultiInputLayer
from bert.layers.layer_norm import LayerNorm

class TransformerDecoderMultiInput(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(TransformerDecoderMultiInput, self).__init__()
        self.layers_num = args.layers_num
        self.layernorm_position = args.layernorm_position
        has_bias = bool(1 - args.remove_transformer_bias)
        self.transformer_decoder = nn.ModuleList([
            TransformerDecoderMultiInputLayer(args) for _ in range(self.layers_num)
        ])
        self.output_dropout = nn.Dropout(args.dropout)
        if self.layernorm_position=="pre":
            self.out_layer_norm = LayerNorm(args.hidden_size, has_bias=has_bias)
    def _init_cache(self):
        self.cache = {}
        for l in range(self.layers_num):
            layer_cache = {
                "art_memory_keys": None,
                "art_memory_values": None,
                "title_memory_keys": None,
                "title_memory_values": None,
                "self_keys": None,
                "self_values": None
            }
            self.cache["layer_{}".format(l)] = layer_cache
    def map_batch_fn(self, fn):
        for layer in self.cache:
            for key in self.cache[layer]:
                self.cache[layer][key] = fn(self.cache[layer][key], dim=0)

    def get_mask(self, memory_bank, src, emb):
        _, src_seq_length, _ = memory_bank.size()
        batch_size, tgt_seq_length, _ = emb.size()
        mask_encoder = (src > 0). \
            unsqueeze(1). \
            repeat(1, tgt_seq_length, 1). \
            unsqueeze(1)
        mask_encoder = mask_encoder.float()
        mask_encoder = (1.0 - mask_encoder).masked_fill(torch.logical_not(mask_encoder), float("-inf"))
        return mask_encoder
    def forward(self, art_encoder_output, title_encoder_output, emb, art, title, extra_mask=None, with_cache=False, st=False):
        """
        Args:
            memory_bank: [batch_size x src_length x hidden_size]
            emb: [batch_size x tgt_length x emb_size]
            src: [batch_size x src_length]
            extra_mask: [1 x num_head x tgt_length x tgt_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        art_mask = self.get_mask(art_encoder_output, art, emb)
        title_mask = self.get_mask(title_encoder_output, title, emb)
        batch_size, tgt_seq_length, _ = emb.size()
        mask_decoder = torch.ones(tgt_seq_length, tgt_seq_length, device=emb.device)
        mask_decoder = torch.tril(mask_decoder)
        mask_decoder = (1.0 - mask_decoder).masked_fill(torch.logical_not(mask_decoder), float("-inf"))
        mask_decoder = mask_decoder.repeat(batch_size, 1, 1, 1)
        if with_cache and extra_mask is not None:
            mask_decoder = extra_mask.repeat(batch_size, 1, 1, 1)
        else:
            if extra_mask is not None:
                mask_decoder = mask_decoder + extra_mask

        hidden = emb

        for i in range(self.layers_num):
            if with_cache:
                hidden = self.transformer_decoder[i](hidden, art_encoder_output, title_encoder_output,mask_decoder, art_mask, title_mask, self.cache["layer_{}".format(i)], st=st)
            else:
                hidden = self.transformer_decoder[i](hidden, art_encoder_output, title_encoder_output, mask_decoder, art_mask, title_mask, st=st)
        if self.layernorm_position == "pre":
            hidden = self.out_layer_norm(hidden)

        return hidden
