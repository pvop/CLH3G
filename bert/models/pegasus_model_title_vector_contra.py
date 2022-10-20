# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from bert.layers.keyword_attn import KeywordAttn
from bert.utils.title_vector_utils import get_segment_avg

class PegasusModelTitleVectorContra(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """

    def __init__(self, args, src_embedding, tgt_embedding, encoder, decoder, target, contra_target, start_id, vocab_size=None, end_id=None):
        super(PegasusModelTitleVectorContra, self).__init__()
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.target = target
        self.contra_target = contra_target
        self.start_id = start_id
        self.vocab = args.vocab
        self.tgt_length = args.tgt_length
        self.min_tgt_length = args.min_tgt_length
        self.beam_size = args.beam_size
        self.bs_group = args.bs_group
        self.pointer_title_vector = args.pointer_title_vector
        self.decoder_title_vector = args.decoder_title_vector
        self.vocab_size = vocab_size
        self.end_id = end_id


    def get_encoder_output(self, src):
        seg = torch.ones_like(src)
        src_emb = self.src_embedding(src, seg=seg)
        encoder_output = self.encoder(src_emb, src)
        return encoder_output

    def forward(self, article, titles, segment_length, mask_tv, mask_contra, tgt):
        tit_encoder_output = self.get_encoder_output(titles)
        tit_encoder_output = tit_encoder_output[:, 0, :]
        title_vector_tv = torch.masked_select(tit_encoder_output, mask_tv).view(-1, tit_encoder_output.size(-1)).contiguous()
        mask_contra = torch.masked_select(tit_encoder_output, mask_contra).view(-1, 2, tit_encoder_output.size(-1)).contiguous()
        loss_contra, _, _ = self.contra_target(mask_contra)
        title_vector = get_segment_avg(title_vector_tv, segment_length.squeeze(1)).unsqueeze(1)
        art_encoder_output = self.get_encoder_output(article)
        art_encoder_output = torch.cat([title_vector, art_encoder_output], dim=1)
        article = torch.cat([torch.ones([article.size(0), 1], device=article.device).long(), article], dim=1)
        tgt_input = torch.cat([torch.ones(tgt.size()[0], 1, dtype=torch.long, device=tgt.device)*self.start_id,
                               tgt[:, 0:-1]], dim=1)
        seg =torch.ones_like(tgt)
        tgt_emb = self.tgt_embedding(tgt_input, seg=seg)
        if self.decoder_title_vector:
            decoder_output = self.decoder(art_encoder_output, tgt_emb, article, st=False)
        else:
            decoder_output = self.decoder(art_encoder_output[:, 1:], tgt_emb, article[:, 1:], st=False)
        if self.pointer_title_vector:
            loss_gen, _, _ = self.target(art_encoder_output[:, 1:], decoder_output, article[:, 1:], tgt, title_vector)
        else:
            loss_gen, _, _ = self.target(art_encoder_output[:, 1:], decoder_output, article[:, 1:], tgt)
        return (loss_gen, loss_contra)

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
    def inference_forward_beam_search(self, article, titles, segment_length, mask_tv, mask_contra):
        batch_size = article.size()[0]
        tit_encoder_output = self.get_encoder_output(titles)
        tit_encoder_output = tit_encoder_output[:, 0, :]
        title_vector_tv = torch.masked_select(tit_encoder_output, mask_tv).view(-1, tit_encoder_output.size(-1)).contiguous()
        title_vector = get_segment_avg(title_vector_tv, segment_length.squeeze(1)).unsqueeze(1)
        art_encoder_output = self.get_encoder_output(article)
        encoder_output = torch.cat([title_vector, art_encoder_output], dim=1)
        article = torch.cat([torch.ones([article.size(0), 1], device=article.device).long(), article], dim=1)
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
            if self.decoder_title_vector:
                decoder_output = self.decoder(encoder_output, tgt_emb, src, with_cache=True)
            else:
                decoder_output = self.decoder(encoder_output[:, 1:], tgt_emb, src[:, 1:], with_cache=True)
            if self.pointer_title_vector:
                logits = self.target.inference_forward(encoder_output[:, 1:], decoder_output, src[:, 1:], encoder_output[:, 0:1])
            else:
                logits = self.target.inference_forward(encoder_output[:, 1:], decoder_output, src[:, 1:])
            # log_probs = torch.log_softmax(logits, dim=-1).squeeze(1)
            log_probs = logits.squeeze(1)
            if step < self.min_tgt_length:
                log_probs[:, self.end_id] = -1e4
            log_probs += topk_log_probs.view(-1).unsqueeze(1)  # (bs*beam_size) *1 * vocab_size
            length_penalty = (step+1) ** 1.5

            curr_scores = log_probs / length_penalty
            cur_len = alive_seq.size(1)

            if (cur_len > 3):
                for i in range(alive_seq.size(0)):
                    fail = False
                    words = [int(w) for w in alive_seq[i]]
                    words = [self.vocab.i2w[w] for w in words]
                    words = ' '.join(words).replace(' ##', '').split()
                    if (len(words) <= 6):
                        continue
                    trigrams = [(words[i - 4], words[i - 3], words[i - 2], words[i - 1], words[i], words[i + 1]) for i
                                in range(4, len(words) - 1)]
                    # trigrams = [(words[i - 1], words[i]) for i in range(1, len(words) - 1)]
                    trigram = tuple(trigrams[-1])
                    if trigram in trigrams[:-1]:
                        fail = True
                    '''if fail:
                        curr_scores[i] = -1e4'''
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
                        # score1, pred1 = best_hpy[1]
                        # score2, pred2 = best_hpy[2]
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