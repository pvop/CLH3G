# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from bert.layers.layer_norm import LayerNorm
from bert.utils.act_fun import gelu


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class ContraTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args):
        super(ContraTarget, self).__init__()
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.mlm_linear_1 = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.mlm_linear_2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.scl = SupConLoss()
        self.contra_lambda = args.contra_lambda

    def mlm(self, memory_bank):
        output_mlm = self.mlm_linear_2(self.mlm_linear_1(memory_bank))
        output_mlm = output_mlm / torch.sqrt(torch.sum(output_mlm ** 2, dim=-1, keepdim=True))
        loss = self.scl(output_mlm)
        return loss, 0, 0

    def forward(self, memory_bank):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]
        Returns:
            loss: Masked language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language model (MLM).
        loss, correct, denominator = self.mlm(memory_bank)
        loss = loss * self.contra_lambda

        return loss, correct, denominator

    def inference_forward(self, memory_bank):
        output_mlm = self.mlm_linear_2(self.mlm_linear_1(memory_bank))
        output_mlm = output_mlm / torch.sqrt(torch.sum(output_mlm ** 2, dim=-1, keepdim=True))
        results = torch.sum(output_mlm[:, 0] * output_mlm[:, 1], dim=-1)
        return results
