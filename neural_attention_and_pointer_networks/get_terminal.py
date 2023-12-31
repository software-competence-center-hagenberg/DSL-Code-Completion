import time
from six.moves import cPickle as pickle
import json
from collections import Counter
import operator

def restore_freq_dict(filename):
    with open(filename, 'rb') as f:
        save = pickle.load(f)
        return  save['freq_dict'], save['terminal_num']


def get_terminal_dict(vocab_size, freq_dict, total_length, verbose=False):
    terminal_dict = dict()
    sorted_freq_dict = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    if verbose:
        for i in range(100):
            print ('the %d frequent terminal: %s, its frequency: %.5f'%(i, sorted_freq_dict[i][0], float(sorted_freq_dict[i][1])/total_length))
    new_freq_dict = sorted_freq_dict[:vocab_size]
    for i, (terminal, frequent) in enumerate(new_freq_dict):
        terminal_dict[terminal] = i
    return terminal_dict, sorted_freq_dict


def save(filename, terminal_dict, terminal_num, sorted_freq_dict, vocab_size):
    with open(filename, 'wb') as f:
        save = {'terminal_dict': terminal_dict,'terminal_num': terminal_num, 'vocab_size': vocab_size, 'sorted_freq_dict': sorted_freq_dict}
        pickle.dump(save, f, protocol=2)

if __name__ == '__main__':
    freq_dict_filename = './data/freq_dict_TTI.pickle'
    target_filename = './data/terminal_dict_TTI.pickle'
    vocab_size = 10
    start_time = time.time()
    freq_dict, terminal_num = restore_freq_dict(freq_dict_filename)
    print(freq_dict['<empty>'], freq_dict['empty'])
    terminal_dict, sorted_freq_dict = get_terminal_dict(vocab_size, freq_dict, True)
    save(target_filename, terminal_dict, terminal_num, sorted_freq_dict, vocab_size)
    print('Finishing generating terminal_dict and takes %.2f'%(time.time() - start_time))