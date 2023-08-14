from unicodedata import bidirectional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=1, device='cuda'):
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        self.device = device
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        one_hot = torch.full((output.shape[1],), self.smoothing_value).to(self.device)
        one_hot[self.ignore_index] = 0
        model_prob = one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        return F.kl_div(output, model_prob, reduction='sum')


class DecoderSimple(nn.Module):
    def __init__(self, hidden_size, vocab_sizeT, vocab_sizeN, embedding_sizeT, embedding_sizeN, dropout, num_layers, device='cuda'):
        super(DecoderSimple, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = dropout
        self.embeddingN = nn.Embedding(vocab_sizeN, embedding_sizeN, vocab_sizeN - 1)
        self.embeddingT = nn.Embedding(vocab_sizeT + 3, embedding_sizeT, vocab_sizeT - 1)
        self.lstm = nn.LSTM(embedding_sizeN + embedding_sizeT, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.w_global = nn.Linear(hidden_size * 3, vocab_sizeT + 3)

    
    def embedded_dropout(self, embedding, words, scale=None):
        dropout = self.dropout
        if dropout > 0:
            mask = embedding.weight.data.new().resize_((embedding.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embedding.weight) / (1 - dropout)
            masked_embed_weight = mask * embedding.weight
        else:
            masked_embed_weight = embedding.weight

        padding_idx = embedding.padding_idx
        if padding_idx is None:
            padding_idx = -1

        words[words >= embedding.weight.size(0)] = padding_idx

        return F.embedding(words, masked_embed_weight, padding_idx, embedding.max_norm, embedding.norm_type, embedding.scale_grad_by_freq, embedding.sparse)


    def forward(self, input, hc, enc_out, mask, h_parent):
        n_input, t_input = input
        batch_size = n_input.size(0)
        n_input = self.embedded_dropout(self.embeddingN, n_input)
        t_input = self.embedded_dropout(self.embeddingT, t_input)
        input = torch.cat([n_input, t_input], 1)
        out, (h, c) = self.lstm(input.unsqueeze(1), hc)
        hidden = h[-1]
        out = out.squeeze(1)
        w_t = F.log_softmax(self.w_global(torch.cat([hidden, out, h_parent], dim=1)), dim=1)
        return w_t, (h, c)


class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, vocab_sizeT, vocab_sizeN, embedding_sizeT, embedding_sizeN, dropout, num_layers, attn_size=50, pointer=True, device='cuda'):
        super(DecoderAttention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.pointer = pointer
        self.device = device
        self.dropout = dropout
        self.embeddingN = nn.Embedding(vocab_sizeN, embedding_sizeN, vocab_sizeN - 1)
        self.embeddingT = nn.Embedding(vocab_sizeT + attn_size + 2, embedding_sizeT, vocab_sizeT - 1)
        self.W_hidden = nn.Linear(hidden_size, hidden_size)
        self.W_mem2hidden = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.W_context = nn.Linear(embedding_sizeN + embedding_sizeT + hidden_size, hidden_size)
        self.lstm = nn.LSTM(embedding_sizeN + embedding_sizeT, hidden_size, num_layers = num_layers, batch_first=True, bidirectional=False)
        self.w_global = nn.Linear(hidden_size * 3, vocab_sizeT + 2)
        if self.pointer:
            self.w_switcher = nn.Linear(hidden_size * 2, 1)
            self.logsigmoid = torch.nn.LogSigmoid()

    
    def embedded_dropout(self, embedding, words, scale=None):
        dropout = self.dropout
        if dropout > 0:
            mask = embedding.weight.data.new().resize_((embedding.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embedding.weight) / (1 - dropout)
            masked_embed_weight = mask * embedding.weight
        else:
            masked_embed_weight = embedding.weight

        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embedding.padding_idx
        if padding_idx is None:
            padding_idx = -1

        words[words >= embedding.weight.size(0)] = padding_idx
        return F.embedding(words, masked_embed_weight, padding_idx, embedding.max_norm, embedding.norm_type, embedding.scale_grad_by_freq, embedding.sparse)

    
    def forward(self, input, hc, enc_out, mask, h_parent):
        n_input, t_input = input
        batch_size = n_input.size(0)

        n_input = self.embedded_dropout(self.embeddingN, n_input)
        t_input = self.embedded_dropout(self.embeddingT, t_input)
        input = torch.cat([n_input, t_input], 1)
        out, (h, c) = self.lstm(input.unsqueeze(1), hc)
        hidden = h[-1]
        out = out.squeeze(1)
        scores = self.W_hidden(hidden).unsqueeze(1)

        if enc_out.shape[1] > 0:
            scores_mem = self.W_mem2hidden(enc_out)
            scores = scores.repeat(1, scores_mem.shape[1], 1) + scores_mem

        scores = torch.tanh(scores)
        scores = self.v(scores).squeeze(2)
        scores = scores.masked_fill(mask, -1e20)
        attn_weights = F.softmax(scores, dim=1)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.matmul(attn_weights, enc_out).squeeze(1)

        if self.pointer:
            w_t = F.log_softmax(self.w_global(torch.cat([context, out, h_parent], dim=1)), dim=1)
            attn_weights = F.log_softmax(scores, dim=1)
            w_s = self.w_switcher(torch.cat([context, out], dim=1))
            return torch.cat([self.logsigmoid(w_s) + w_t, self.logsigmoid(-w_s) + attn_weights], dim=1), (h, c)
        else:
            w_t = F.log_softmax(self.w_global(torch.cat([context, out, h_parent], dim=1)), dim=1)
            return w_t, (h, c)


class MixtureAttention(nn.Module):
    def __init__(self, hidden_size, vocab_sizeT, vocab_sizeN, embedding_sizeT, embedding_sizeN, num_layers, dropout, device='cuda', label_smoothing=0.1, attn=True, pointer=True, attn_size=50, SOS_token=0):
        super(MixtureAttention, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.eof_N_id = vocab_sizeN - 1
        self.eof_T_id = vocab_sizeT - 1
        self.unk_id = vocab_sizeT - 2
        self.SOS_token = SOS_token
        self.attn_size = attn_size
        self.vocab_sizeT = vocab_sizeT
        self.vocab_sizeN = vocab_sizeN
        self.W_out = nn.Linear(hidden_size * 2, hidden_size)
        
        if attn:
            self.decoder = DecoderAttention(hidden_size, vocab_sizeT, vocab_sizeN, embedding_sizeT, embedding_sizeN, dropout, num_layers, attn_size, pointer, device).to(device)
        else:
            self.decoder = DecoderSimple(hidden_size, vocab_sizeT, vocab_sizeN, embedding_sizeT, embedding_sizeN, dropout, num_layers, device).to(device)

        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(label_smoothing, vocab_sizeT + attn_size + 3, self.eof_T_id, device)
        else:
            self.criterion = nn.NLLLoss(reduction='none', ignore_index=self.eof_T_id)

        self.pointer = pointer

    def forward(self, n_tensor, t_tensor, p_tensor):
        batch_size = n_tensor.size(0)
        max_length = n_tensor.size(1)

        full_mask = (n_tensor == self.eof_N_id)

        input = (torch.ones(batch_size, dtype=torch.long, device=self.device) * self.SOS_token,
                 torch.ones(batch_size, dtype=torch.long, device=self.device) * self.SOS_token)

        hs = torch.zeros(batch_size, max_length, self.hidden_size, requires_grad=False).to(self.device)
        hc = None

        parent = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        token_losses = torch.zeros(batch_size, max_length).to(self.device)

        ans = []

        for iter in range(max_length):
            memory = hs[:, max(iter - self.attn_size, 0) : iter]
            output, hc = self.decoder(input, hc, memory.clone().detach(), full_mask[:, max(iter - self.attn_size, 0) : iter], hs[torch.arange(batch_size),parent].squeeze(1).clone().detach())
            hs[:, iter] = hc[0][-1]
            topv, topi = output.topk(1)
            input = (n_tensor[:, iter].clone(), t_tensor[:, iter].clone())
            parent = p_tensor[:, iter]
            ans.append(topi.detach())
            target = t_tensor[:, iter]
            target[target >= output.shape[1]] = self.unk_id
            token_losses[:, iter] = self.criterion(output, t_tensor[:, iter].clone().detach())

        loss = token_losses.sum()
        return loss, torch.cat(ans, dim=1)
