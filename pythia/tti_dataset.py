# https://git.coinse.io/coinse/code2seq-get-embedding/-/blob/4e93f27e72958b9e081083a01b3459f31a69cd05/Python150kExtractor/extract.py

import os
import torch
import torch.utils.data as data
import numpy as np
import csv

UNKNOWN_TOKEN = '<UNK>'
PADDING_TOKEN = '<PAD>'

def add_unknown_token(vocab):
    return np.append(vocab, UNKNOWN_TOKEN)

def add_padding_token(vocab):
    return np.append(vocab, PADDING_TOKEN)

def create_type_mapping(vocab):
    type2idx = {}
    idx2type = {}

    for i, w in enumerate(vocab):
        t = w.split(':')
        if len(t) > 1:
            type = t[1]
            idx2type[i] = type

            if type in type2idx:
                type2idx[type].append(i)
            else:
                type2idx[type] = [i]
        else:
            idx2type[i] = None

    return type2idx, idx2type


def create_mappings(vocab):
    word2idx = {w: idx for (idx, w) in enumerate(vocab)}
    idx2word = {idx: w for (idx, w) in enumerate(vocab)}
    type2idx, idx2type = create_type_mapping(vocab)
    return word2idx, idx2word, type2idx, idx2type


def get_files(folder):
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    listed_files = {}

    for path, _, files in os.walk(folder):
        for file in files:
            id = file.split(sep='.')[0]
            full_path = os.path.join(path, file)
            listed_files[int(id)] = full_path

    return listed_files


class TtiDataset(data.Dataset):
    train_dir = 'training'
    val_dir = 'validation'
    test_dir = 'test'
    vocab_file = 'voc.npy'

    def __init__(self, root_dir, mode='train', lookback_tokens=100, chunk_size=100, max_len_label=6):
        self.root_dir = root_dir
        self.mode = mode
        self.idx_cached = -1
        self.lookback_tokens = lookback_tokens
        self.max_len_label = max_len_label
        self.vocab = self.init_vocab(root_dir)
        self.word2idx, self.idx2word, self.type2idx, self.idx2type = create_mappings(self.vocab)
        self.padding_idx = self.word2idx[PADDING_TOKEN]
        self.chunk_size = chunk_size

        if self.mode.lower() == 'train':
            self.files = get_files(os.path.join(root_dir, self.train_dir))
        elif self.mode.lower() == 'val':
            self.files = get_files(os.path.join(root_dir, self.val_dir))
        elif self.mode.lower() == 'test':
            self.files = get_files(os.path.join(root_dir, self.test_dir))
        else:
            raise RuntimeError("Unsupported dataset mode. Supported modes are: train, val and test")


    def init_vocab(self, root_dir):
        vocab = np.load(os.path.join(root_dir, self.vocab_file))
        vocab = add_unknown_token(vocab)
        vocab = add_padding_token(vocab)
        return vocab

    
    def gen_from_txt(self, csvfile):
        data = None
        with open(csvfile) as f:
            data = list(csv.reader(f, delimiter=','))
            data = np.array(data, dtype=np.float)
        return data


    def __getitem__(self, index):
        file_idx = np.floor(index/self.chunk_size)

        if self.idx_cached != file_idx:
            self.data_cached = self.gen_from_txt(self.files[file_idx])
            self.idx_cached = file_idx
        
        row = int(index - (self.chunk_size * file_idx))
        data_len = self.data_cached[row][0]
        padding_start = int(data_len) + 1
        data_unpadded = self.data_cached[row][1:padding_start]
        padding = self.data_cached[row][padding_start:self.lookback_tokens+1]
        data = np.concatenate([data_unpadded, padding])
        label_data = self.data_cached[row][self.lookback_tokens + 1:]
        label_len = len(label_data)
        return torch.LongTensor(data), data_len, torch.LongTensor(label_data), label_len


    def __len__(self):
        if self.mode.lower() == 'train':
            return 1607
        elif self.mode.lower() == 'val':
            return 689
        elif self.mode.lower() == 'test':
            return 911213
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def get_vocab_len(self):
        return len(self.vocab)

    def get_mappings(self):
        return self.word2idx, self.idx2word, self.type2idx, self.idx2type