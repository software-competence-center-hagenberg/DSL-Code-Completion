import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.special import softmax
import numpy as np
from queue import PriorityQueue
from functools import total_ordering
import torch.optim as optim
from tti_dataset import TtiDataset
from torch.utils.data import DataLoader
import time
from torch.nn.utils import clip_grad_norm_
import argparse


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@total_ordering
class Sample:
    def __init__(self, s: str, indices, logp: float, state):
        self.s = s
        self.indices = indices
        self.logp = logp
        self.state = state
                
    def __lt__(self, other):
        self.logp < other.logp

    def __eq__(self, other):
        self.logp == other.logp


class NeuralCodeCompletion(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, batch_size, num_layers, dropout, padding_idx=None):
        super(NeuralCodeCompletion, self).__init__()
        self.padding_idx = padding_idx
        if padding_idx is None:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.output_dim = embedding_dim
        self.linear = nn.Linear(hidden_dim, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, vocab_size)


    def forward(self, inputs, input_len):
        embeddings = self.embeddings(inputs)
        embeddings = pack_padded_sequence(embeddings, input_len, batch_first=True, enforce_sorted=False)
        lstm_packed_out, (ht, _) = self.LSTM(embeddings)
        _, _ = pad_packed_sequence(lstm_packed_out, padding_value=self.padding_idx)
        return self.decoder(self.linear(ht[-1]))
    

    def beam_search_forward(self, inputs, input_len, vocab, beam_width=3):
        topK = 10
        decoded_batch = []
        inputs.detach()

        for i in range(inputs.size(0)):
            embeddings = self.embeddings(inputs[i:i+1])
            embeddings = pack_padded_sequence(embeddings, input_len[i:i+1], batch_first=True, enforce_sorted=False)

            _, (h0, c0) = self.lstm(embeddings)
            top_10 = PriorityQueue(topK)
            candidates = PriorityQueue()
            pred_embed = self.linear(h0[-1])
            projection = self.decoder(pred_embed)
            probs = np.log(softmax(projection.detach().numpy().flatten()))
            best = np.argsort(probs)[::-1]
            for _, i in zip(range(beam_width), best):
                candidates.put(Sample(str(vocab[i]), [i], probs[i], (h0, c0)), block=False)

            while not candidates.empty():
                candidate = candidates.get(block=False)
                embed = self.embeddings(torch.Tensor([[candidate.indices[-1]]]).to(torch.long))
                lstm_packed_out, (ht, ct) = self.lstm(embed, candidate.state)
                pred_embed = self.linear(ht[-1])
                projection = self.decoder(pred_embed)
                probs = np.log(softmax(projection.detach().numpy().flatten()))
                best = np.argsort(probs)[::-1]
                for _, i in zip(range(beam_width), best):
                    candidates.put(Sample(candidate.s + " " + str(vocab[i]), candidate.indices + [i], candidate.logp + probs[i], (ht, ct)), block=False)
                    print(candidate.s + " " + str(vocab[i]))


def main():
    parser = argparse.ArgumentParser(description="Hyperparamters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--embedding_dim", help="embedding dimension")
    parser.add_argument("--batch_size", help="batch size")
    parser.add_argument("--num_layers", help="number of recurrent layers")
    parser.add_argument("--hidden_dim", help="hidden dimension")
    parser.add_argument("--dropout", help="dropout")
    parser.add_argument("--lr_init", help="initial learning rate")
    parser.add_argument("--weight_decay", help="weight decay per epoch")
    parser.add_argument("--l2_regularization", help="l2 regularization")
    parser.add_argument("--lookback", help="number of lookback tokens")
    parser.add_argument("--num_workers", help="number of threads")
    parser.add_argument("--epochs", help="number of epochs")
    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    lr_init = args.lr_init
    weight_decay_per_epoch = args.weight_decay
    l2_regularization = args.l2_regularization
    dataset_path = './data'
    lookback_tokens = args.lookback
    pin_memory = True
    num_workers = args.num_workers
    epochs = args.epochs
    torch.autograd.set_detect_anomaly(True)

    trainset = TtiDataset(dataset_path, mode='train', lookback_tokens=lookback_tokens)
    valset = TtiDataset(dataset_path, mode='val', lookback_tokens=lookback_tokens)

    dataloaders = { 'train': DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers), 'val': DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers) }
    model = NeuralCodeCompletion(embedding_dim, trainset.get_vocab_len(), hidden_dim, batch_size, num_layers, dropout, trainset.padding_idx)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=trainset.padding_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=l2_regularization)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=weight_decay_per_epoch)
    word2idx, idx2word, type2dix, idx2type = trainset.get_mappings()
    best_model = None
    best_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc, train_acc_5 = train_epoch(dataloaders['train'], model,
                                            criterion, optimizer, scheduler, epoch, type2dix, idx2type,
                                            trainset.get_vocab_len())
        print('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch, train_loss, train_acc))

        val_loss, val_acc, val_acc_5 = validate_epoch(dataloaders['val'], model,
                                            criterion, epoch, type2dix, idx2type,
                                            trainset.get_vocab_len())

        print('Epoch {} validation loss: {:.4f}, acc: {:.4f}'.format(epoch, val_loss, val_acc))

        if val_acc_5 > best_acc:
            best_model = model.state_dict()


        


def calc_acc(input, label, input_len, out, type2idx, idx2type):
    """Calculates accuracy based on the matching types."""
    top_1_correct = 0
    top_5_correct = 0
    for i, sample in enumerate(input):
        type = idx2type[int(label[i])]
        last_token_type = idx2type[int(sample[int(input_len[i])-2])]

        # Type of last token doesn't match type of label
        if type is None or last_token_type != type:
            index = out[i].argmax()
            if index == label.data[i]:
                top_1_correct += 1

            values, indices = out[i].topk(k=5)
            if label.data[i] in indices:
                top_5_correct += 1

        # Type of last token matches type of label
        else:
            out_narrowed = out[i][type2idx[type]]
            index = out_narrowed.argmax()
            original_index = type2idx[type][index]
            if original_index == label.data[i]:
                top_1_correct += 1

            k = 5
            l = len(out_narrowed)
            if l < 5:
                k = l
            values, indices = out_narrowed.topk(k=k)
            original_indices = [type2idx[type][i] for i in indices]
            if label.data[i] in original_indices:
                top_5_correct += 1

    return top_1_correct, top_5_correct



def train_epoch(dataloader, model, criterion, optimizer, lr_scheduler, epoch, type2idx, idx2type, vocab_len):
    max_norm_grad = 5
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    top1_acc_running = AverageMeter('Top-1 Accuracy', ':.3f')
    top5_acc_running = AverageMeter('Top-5 Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, top1_acc_running, top5_acc_running],
        prefix="Train, epoch: [{}]".format(epoch))

    end = time.time()

    model.train()

    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (input, input_len, label, label_len) in enumerate(dataloader):

            optimizer.zero_grad()

            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()

            out = model(input, input_len)



            # Ignore prediction for padding token
            out = out[..., :vocab_len-1].contiguous()

            # Optimizer
            label = torch.max(label, 1)[1]
            loss = criterion(out, label)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm_grad)
            optimizer.step()

            top1_corrects, top5_corrects = calc_acc(input, label, input_len, out, type2idx, idx2type)

            # Statistics
            bs = input.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)

            # Top-1 accuracy
            top1_acc = top1_corrects / bs
            top1_acc_running.update(top1_acc, bs)

            # Top-5 accuracy
            top5_acc = top5_corrects / bs
            top5_acc_running.update(top5_acc, bs)

            # output training info
            progress.display(epoch_step)

            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

        # Reduce learning rate
        lr_scheduler.step()

    return loss_running.avg, top1_acc_running.avg, top5_acc_running.avg


def validate_epoch(dataloader, model, criterion, epoch, idx2word, idx2type, vocab_len):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Top-1 Accuracy', ':.3f')
    top5_acc_running = AverageMeter('Top-5 Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running, top5_acc_running],
        prefix="Validate, epoch: [{}]".format(epoch))

    model.eval()

    with torch.no_grad():
        end = time.time()

        for epoch_step, (input, input_len, label, label_len) in enumerate(dataloader):
            data_time.update(time.time() - end)

            if (torch.cuda.is_available()):
                input = input.cuda()
                label = label.cuda()

            out = model(input, input_len)

            # Ignore prediction for padding token            
            out = out[..., :vocab_len-1].contiguous()
            label = torch.max(label, 1)[1]
            loss = criterion(out, label)

            # Statistics
            bs = input.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)

            top1_corrects, top5_corrects = calc_acc(input, label, input_len, out, idx2word, idx2type)

            # Top-1 accuracy
            acc = top1_corrects / bs
            acc_running.update(acc, bs)

            # Top-5 accuracy
            top5_acc = top5_corrects / bs
            top5_acc_running.update(top5_acc, bs)

            # output training info
            progress.display(epoch_step)

             # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

    return loss_running.avg, acc_running.avg, top5_acc_running.avg


if __name__ == "__main__":
    main()