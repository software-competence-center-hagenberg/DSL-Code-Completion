import torch
import time
from six.moves import cPickle as pickle
import torch.utils.data as data

def fix_parent(p, start_i):
    p -= start_i
    return 0 if p < 0 else p


def data_gen(data, split_size):
    for sample in data:
        accum_n = []
        accum_t = []
        accum_p = []
        start_i = 0

        for i, item in enumerate(zip(*sample)):
            n, t, p = item
            p = fix_parent(p, start_i)
            accum_n.append(n)
            accum_t.append(t)
            accum_p.append(p)

            if len(accum_n) == split_size:
                yield accum_n, accum_t, accum_p
                accum_n = []
                accum_t = []
                accum_p = []
                start_i = i

        if len(accum_n) > 0:
            yield accum_n, accum_t, accum_p


def input_data(N_filename, T_filename):
    start_time = time.time()
    with open(N_filename, 'rb') as f:
        print ("reading data from ", N_filename)
        save = pickle.load(f)
        train_dataN = save['trainData']
        test_dataN = save['testData']
        train_dataP = save['trainParent']
        test_dataP = save['testParent']
        vocab_sizeN = save['vocab_size']
        print ('the vocab_sizeN is %d (not including the eof)' %vocab_sizeN)
        print ('the number of training data is %d' %(len(train_dataN)))
        print ('the number of test data is %d\n' %(len(test_dataN)))

    with open(T_filename, 'rb') as f:
        print ("reading data from ", T_filename)
        save = pickle.load(f)
        train_dataT = save['trainData']
        test_dataT = save['testData']
        vocab_sizeT = save['vocab_size']
        attn_size = save['attn_size']
        print ('the vocab_sizeT is %d (not including the unk and eof)' %vocab_sizeT)
        print ('the attn_size is %d' %attn_size)
        print ('the number of training data is %d' %(len(train_dataT)))
        print ('the number of test data is %d' %(len(test_dataT)))
        print ('Finish reading data and take %.2f\n'%(time.time()-start_time))

    return train_dataN, test_dataN, vocab_sizeN, train_dataT, test_dataT, vocab_sizeT, attn_size, train_dataP, test_dataP


class TtiDataset(data.Dataset):
    def __init__(self, N_filename, T_filename, is_train=False, truncate_size=150):
        super(TtiDataset).__init__()
        train_dataN, test_dataN, vocab_sizeN, train_dataT, test_dataT, vocab_sizeT, attn_size, train_dataP, test_dataP = input_data(N_filename, T_filename)
        self.is_train = is_train
        
        if self.is_train:
            self.data = [item for item in data_gen(zip(train_dataN, train_dataT, train_dataP), truncate_size)]
        else:
            self.data = [item for item in data_gen(zip(test_dataN, test_dataT, test_dataP), truncate_size)]

        self.data = sorted(self.data, key=lambda x: len(x[0]))
        self.vocab_sizeN = vocab_sizeN
        self.vocab_sizeT = vocab_sizeT
        self.attn_size = attn_size
        self.eof_N_id = vocab_sizeN - 1
        self.eof_T_id = vocab_sizeT - 1
        self.unk_id = vocab_sizeT - 2
        self.truncate_size = truncate_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, samples, device='cpu'):
        sent_N = [sample[0] for sample in samples]
        sent_T = [sample[1] for sample in samples]
        sent_P = [sample[2] for sample in samples]

        s_max_length = max(map(lambda x: len(x), sent_N))

        sent_N_tensors = []
        sent_T_tensors = []
        sent_P_tensors = []

        for sn, st, sp in zip(sent_N, sent_T, sent_P):
            sn_tensor = torch.ones(s_max_length, dtype=torch.long, device=device) * self.eof_N_id
            st_tensor = torch.ones(s_max_length, dtype=torch.long, device=device) * self.eof_T_id
            sp_tensor = torch.ones(s_max_length, dtype=torch.long, device=device) * 1

            for idx, w in enumerate(sn):
                sn_tensor[idx] = w
                st_tensor[idx] = st[idx]
                sp_tensor[idx] = sp[idx]

            sent_N_tensors.append(sn_tensor.unsqueeze(0))
            sent_T_tensors.append(st_tensor.unsqueeze(0))
            sent_P_tensors.append(sp_tensor.unsqueeze(0))

        sent_N_tensors = torch.cat(sent_N_tensors, dim=0)
        sent_T_tensors = torch.cat(sent_T_tensors, dim=0)
        sent_P_tensors = torch.cat(sent_P_tensors, dim=0)
        return sent_N_tensors, sent_T_tensors, sent_P_tensors
