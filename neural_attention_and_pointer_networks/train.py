import torch
from tti_dataset import TtiDataset
from model import MixtureAttention
import numpy as np
from sklearn.metrics import accuracy_score
import argparse


try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboard import SummaryWriter

def accuracy(out, target, ignored_index, unk_index):
    out_ = np.array(out[target != ignored_index])
    target_ = np.array(target[target != ignored_index])
    out_[out_ == unk_index] = -1
    return accuracy_score(out_, target_)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def train():
    writer = SummaryWriter('logs/')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    N_filename = "./data/TTI_non_terminal.pickle"
    T_filename = "./data/TTI_terminal_whole.pickle"
    parser = argparse.ArgumentParser(description="Hyperparamters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--truncate_size", help="truncate size")
    parser.add_argument("--batch_size", help="batch size")
    parser.add_argument("--num_layers", help="number of recurrent layers")
    parser.add_argument("--hidden_dim", help="hidden dimension")
    parser.add_argument("--dropout", help="dropout")
    parser.add_argument("--lr", help="learning rate")
    parser.add_argument("--lr_decay", help="learning rate")
    parser.add_argument("--weight_decay", help="weight decay per epoch")
    parser.add_argument("--num_workers", help="number of threads")
    parser.add_argument("--epochs", help="number of epochs")
    parser.add_argument("--embedding_sizeT", help="embedding size terminal")
    parser.add_argument("--embedding_sizeN", help="embedding size non terminal")
    parser.add_argument("--label_smoothing", help="label smoothing")
    parser.add_argument('--pointer', action='store_true', default=False)
    parser.add_argument("--attn", help="Attention")
    parser.add_argument("--train_clip", help="train clip")
    args = parser.parse_args()
    truncate_size = args.truncate_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    hidden_size = args.hidden_dim
    embedding_sizeT = args.embedding_sizeT
    embedding_sizeN = args.embedding_sizeN
    num_layers = args.num_layers
    dropout = args.dropout
    label_smoothing = args.label_smoothing
    pointer = args.pointer
    attn = args.attn
    lr = args.lr
    lr_decay = args.lr_decay
    train_clip = args.train_clip


    data_train = TtiDataset(N_filename, T_filename, True, truncate_size)
    data_val = TtiDataset(N_filename, T_filename, True, truncate_size)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=data_train.collate_fn)
    test_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=data_val.collate_fn)
    ignored_index = data_train.vocab_sizeT - 1
    unk_index = data_train.vocab_sizeT - 2

    model = MixtureAttention(hidden_size, data_train.vocab_sizeT, data_train.vocab_sizeN, embedding_sizeT, embedding_sizeN, num_layers, dropout, device, label_smoothing, True, pointer, attn)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(0, epochs):
        lr = lr * lr_decay ** max(epoch - 1, 0)
        adjust_learning_rate(optimizer, lr)
        print("epoch: %04d" % epoch)
        loss_avg, acc_avg = 0, 0
        total = len(train_loader)
        model = model.train()

        for i, (n, t, p) in enumerate(train_loader):
            n, t, p = n.to(device), t.to(device), p.to(device)
            optimizer.zero_grad()

            loss, ans = model(n, t, p)
            loss_avg += loss.item()
            acc_item = accuracy(ans.cpu().numpy().flatten(), t.cpu().numpy().flatten(), ignored_index, unk_index)
            acc_avg += acc_item
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_clip)
            loss.backward()

            if (i + 1) % 100 == 0:
                print('\ntemp_loss: %f, temp_acc: %f' % (loss.item(), acc_item), flush=True)
                writer.add_scalar('train/loss', loss.item(), epoch * total + i)
                writer.add_scalar('train/acc', acc_item, epoch * total + i)

            optimizer.step()

        print("\navg_loss: %f, avg_acc: %f" % (loss_avg/total, acc_avg/total))

        with torch.no_grad():
            model = model.eval()
            acc = 0
            loss_eval = 0

            for i, (n, t, p) in enumerate(test_loader):
                n, t, p = n.to(device), t.to(device), p.to(device)
                loss, ans = model(n, t, p)
                loss_eval += loss.item()
                acc += accuracy(ans.cpu().numpy().flatten(), t.cpu().numpy().flatten(), ignored_index, unk_index)

            l = len(test_loader)
            acc /= l
            loss_eval /= l
            print('\navg acc:', acc, 'avg loss:', loss_eval)
            writer.add_scalar('val/loss', loss_eval, epoch)
            writer.add_scalar('val/acc', acc, epoch)


if __name__ == '__main__':
    train()