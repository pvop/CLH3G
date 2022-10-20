# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from bert.layers.layer_norm import LayerNorm
from bert.utils.act_fun import gelu
import torch.nn.functional as F


class PegasusTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM) 
    and next sentence prediction (NSP) for pretraining.
    """
    def __init__(self, args, vocab_size):
        super(PegasusTarget, self).__init__()
        self.vocab_size = vocab_size
        self.num = 0
        self.hidden_size = args.hidden_size
        self.sparse_num = self.vocab_size
        self.copy_num = args.seq_length
        self.emb_size = args.emb_size
        has_bias = bool(1 - args.remove_transformer_bias)
        self.pointer_atten_1 = nn.Linear(args.hidden_size, args.hidden_size, bias=has_bias)
        self.pointer_atten_2 = nn.Linear(args.hidden_size, args.hidden_size, bias=has_bias)
        self.alpha_prob_linear = nn.Linear(2 * args.hidden_size, 1, bias=has_bias)
        self.mlm_linear_1 = nn.Linear(args.hidden_size, self.vocab_size, bias=has_bias)

        self.softmax = nn.LogSoftmax(dim=-1)



        self.criterion = nn.NLLLoss()
    def pgn_inference(self, encoder_outputs, decoder_outputs, src_input):
        tgt_len = decoder_outputs.size(1)
        pointer_weights = torch.matmul(self.pointer_atten_1(decoder_outputs),
                                       self.pointer_atten_2(encoder_outputs).transpose(1, 2)).squeeze(2)
        pointer_prob = torch.softmax(pointer_weights, dim=2)
        encoder_atten_vector = torch.matmul(pointer_prob, encoder_outputs)
        alpha_prob = F.sigmoid(self.alpha_prob_linear(torch.cat([decoder_outputs, encoder_atten_vector], dim=-1)))
        vocab_prob = torch.softmax(self.mlm_linear_1(decoder_outputs), dim=-1)
        vocab_prob = alpha_prob * vocab_prob
        pointer_prob = (1 - alpha_prob) * pointer_prob
        src_input = src_input.unsqueeze(1).repeat(1, tgt_len, 1)
        vocab_prob = vocab_prob.scatter_add_(2, src_input, pointer_prob)
        return vocab_prob

    def pgn_mlm(self, encoder_outputs, decoder_outputs, src_input):
        vocab_prob = torch.softmax(self.mlm_linear_1(decoder_outputs), dim=-1)
        return vocab_prob

    def mlm(self, encoder_outputs, decoder_outputs, src_input, tgt_mlm):
        output_mlm = self.pgn_inference(encoder_outputs, decoder_outputs, src_input)
        #output_mlm = self.pgn_mlm(encoder_outputs, decoder_outputs, src_input)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm.view(-1, output_mlm.size(-1))
        output_mlm = output_mlm[tgt_mlm>0]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        one_hot = torch.zeros(output_mlm.size(0),  self.vocab_size). \
           to(torch.device(output_mlm.device)). \
           scatter_(1, tgt_mlm.contiguous().view(-1,1), 1.0)
        numerator = -torch.sum(torch.log(output_mlm+1e-20) * one_hot, 1)
        denominator = torch.tensor(output_mlm.size(0) + 1e-4)
        loss_mlm = torch.sum(numerator) / denominator





        if output_mlm.size(0) == 0:
            correct_mlm = torch.tensor(0.0)
        else:
            correct_mlm = torch.sum((output_mlm.argmax(dim=-1).eq(tgt_mlm)).float())
        
        return loss_mlm, correct_mlm, denominator

    def forward(self, encoder_outputs, decoder_outputs, src_input, tgt_mlm):
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
        loss, correct, denominator = self.mlm(encoder_outputs, decoder_outputs, src_input, tgt_mlm)

        return loss, correct, denominator
    def inference_forward(self, encoder_outputs, decoder_outputs, src_input):
        output_mlm = self.pgn_inference(encoder_outputs, decoder_outputs, src_input)
        output_mlm = torch.log(output_mlm + 1e-20)
        return output_mlm
