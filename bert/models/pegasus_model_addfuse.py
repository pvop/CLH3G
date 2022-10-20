# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from bert.layers.keyword_attn import KeywordAttn

class PegasusModelAddFuse(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """

    def __init__(self, args, src_embedding, tgt_embedding, encoder, decoder, target, start_id, vocab_size=None, end_id=None):
        super(PegasusModelAddFuse, self).__init__()
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.target = target
        self.start_id = start_id
        self.vocab = args.vocab
        self.tgt_length = args.tgt_length
        self.min_tgt_length = args.min_tgt_length
        self.beam_size = args.beam_size
        self.bs_group = args.bs_group
        self.vocab_size = vocab_size
        self.end_id = end_id
        self.key_attn = KeywordAttn(args)

    def get_encoder_output(self, src):
        seg = torch.ones_like(src)
        src_emb = self.src_embedding(src, seg=seg)
        encoder_output = self.encoder(src_emb, src)
        return encoder_output
    def forward(self, article, titles, tgt):
        art_encoder_output = self.get_encoder_output(article)
        tit_encoder_output = self.get_encoder_output(titles)
        encoder_output = self.key_attn(art_encoder_output, tit_encoder_output, article, titles)
        tgt_input = torch.cat([torch.ones(tgt.size()[0], 1, dtype=torch.long, device=tgt.device)*self.start_id,
                               tgt[:, 0:-1]], dim=1)
        seg =torch.ones_like(tgt)
        tgt_emb = self.tgt_embedding(tgt_input, seg=seg)
        decoder_output = self.decoder(encoder_output, tgt_emb, article, st=False)
        loss_info = self.target(encoder_output, decoder_output, article, tgt)
        return loss_info

    def tile(self, x, count, dim=0):
        """
        Tiles x on dimension dim count times.
        """
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x
    def inference_forward_beam_search(self, article, titles):
        batch_size = article.size()[0]
        art_encoder_output = self.get_encoder_output(article)
        tit_encoder_output = self.get_encoder_output(titles)
        encoder_output = self.key_attn(art_encoder_output, tit_encoder_output, article, titles)
        src = self.tile(article, self.beam_size, dim=0)
        encoder_output = self.tile(encoder_output, self.beam_size, dim=0)
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=src.device)
        beam_offset = torch.arange(
            0,
            batch_size * self.beam_size,
            step=self.beam_size,
            dtype=torch.long,
            device=src.device
        )
        alive_seq = torch.full([batch_size * self.beam_size, 1],
                               self.start_id,
                               dtype=torch.long,
                               device=src.device)
        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (self.beam_size - 1), device=src.device).repeat(batch_size)
        )
        # Structure that holds finished hypotheses
        hypotheses = [[] for _ in range(batch_size)]

        results = [[] for _ in range(batch_size)]
        self.decoder._init_cache()
        for step in range(self.tgt_length):
            decoder_input = alive_seq[:, -1].view(-1, 1)
            pos = torch.ones_like(decoder_input, device=decoder_input.device) * step
            seg = torch.ones_like(decoder_input)
            # decoder forward
            tgt_emb = self.tgt_embedding(decoder_input, pos=pos, seg=seg)
            decoder_output = self.decoder(encoder_output, tgt_emb, src, with_cache=True)
            logits = self.target.inference_forward(encoder_output, decoder_output, src)
            # log_probs = torch.log_softmax(logits, dim=-1).squeeze(1)
            log_probs = logits.squeeze(1)
            if step < self.min_tgt_length:
                log_probs[:, self.end_id] = -1e4
            log_probs += topk_log_probs.view(-1).unsqueeze(1)  # (bs*beam_size) *1 * vocab_size
            length_penalty = (step+1) ** 1.5

            curr_scores = log_probs / length_penalty

            
            for i in range(alive_seq.size(0)):
                if alive_seq[i][-1] == self.end_id:
                    curr_scores[i] = -1e4

            curr_scores = curr_scores.reshape(-1, self.beam_size * self.vocab_size)
            topk_scores, topk_ids = curr_scores.topk(self.beam_size, dim=-1)
            # Recover log probs because we do only once for a single sequence
            topk_log_probs = topk_scores * length_penalty

            topk_beam_index = topk_ids.div(self.vocab_size).long()  # (bs*beam_size)
            topk_ids = topk_ids.fmod(self.vocab_size)

            # map beam_index to batch_index
            batch_index = (topk_beam_index + beam_offset[:topk_beam_index.size()[0]].unsqueeze(1))

            select_indices = batch_index.view(-1)

            # Append last prediction
            alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], dim=-1)

            is_finished = topk_ids.eq(self.end_id)
            if step + 1 == self.tgt_length:
                is_finished.fill_(1)
            end_condition = is_finished[:, 0].eq(1)
            if is_finished.any():
                predictions = alive_seq.view(-1, self.beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                    if end_condition[i]:
                        best_hpy = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hpy[0]
                        results[b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                if len(non_finished) == 0:
                    break

                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))
            select_indices = batch_index.view(-1)
            encoder_output = encoder_output.index_select(0, select_indices)
            src = src.index_select(0, select_indices)
            self.decoder.map_batch_fn(lambda state, dim: state.index_select(dim, select_indices))
        self.decoder.cache = None
        return results