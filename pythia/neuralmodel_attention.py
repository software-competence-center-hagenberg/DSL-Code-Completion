import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1.0 / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        query = query.unsqueeze(1)
        keys = keys.transpose(0, 1).transpose(1, 2)
        energy = torch.bmm(query, keys)
        energy = nn.functional.softmax(energy.mul_(self.scale), dim=2)
        values = values.transpose(0, 1)
        linear_combination = torch.bmm(energy, values).squeeze(1)
        return energy, linear_combination

class Attention_custom(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention_custom, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.attn_bpe = nn.Linear(150*2, 150)
        self.v_bpe = nn.Parameter(torch.rand(150))
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, encoder_out, bpe=False):
        max_len = encoder_out.size(0)
        batch_size = encoder_out.size(0)
        H = inputs.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_out = encoder_out.transpose(0, 1)

        if bpe == True:
            attn_weights = self.score_bpe(H, encoder_out)
        else:
            attn_weights = self.score(H, encoder_out)
        
        attn_weights = nn.functional.softmax(attn_weights).unsqueeze(1)
        return attn_weights.bmm(encoder_out)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class NeuralCodeCompletionAttention(nn.Module):
    def __init__(self, embedding_dim, vocab_size, padding_idx, hidden_dim, batch_size, num_layers, dropout):
        super(NeuralCodeCompletionAttention, self).__init__()
        self.padding_idx = padding_idx
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(self.hidden_dim)
        self.attention_custom = Attention_custom(self.hidden_dim)
        self.output_dim = embedding_dim
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs, inputs_len):
        embeddings = self.embeddings(inputs)
        embeddings = pack_padded_sequence(embeddings, inputs_len, batch_first=True, enforce_sorted=False)
        lstm_packed_out, (ht, ct) = self.lstm(embeddings)
        unpacked_output, input_sizes = pad_packed_sequence(lstm_packed_out, padding_value=self.padding_idx)
        hidden = ht[-1]
        linear_combination = self.attention_custom(hidden, unpacked_output)
        pred_embed = self.linear(linear_combination)
        return torch.matmul(self.embeddings.weight, pred_embed.squeeze(1).permute(1, 0)).permute(1, 0)
