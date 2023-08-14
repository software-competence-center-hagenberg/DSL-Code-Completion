import numpy as np
from six.moves import cPickle as pickle
import json
from collections import deque
import time
import os


def restore_terminal_dict(filename):
  with open(filename, 'rb') as f:
    save = pickle.load(f)
    return save['terminal_dict'], save['terminal_num'], save['vocab_size']


def process_binary_circuit(json_dict, terminals, attn_que, unk_id, attn_success_cnt, attn_fail_cnt):
    if json_dict['LeftHandSide'] == 'BinaryCircuitExpression':
        terminals.append(terminal_dict['<empty>'])
        attn_que.append('<empty>')
        return process_binary_circuit(json_dict['LeftHandSide'], terminals, attn_que, unk_id, attn_success_cnt, attn_fail_cnt)
    else:
        type = json_dict['Type']

        if type in terminal_dict.keys():
            terminals.append(terminal_dict[type])
            attn_que.append('Normal')
        else:
            if type in attn_que:
                location_index = [len(attn_que)-ind for ind,x in enumerate(attn_que) if x==type][-1]
                location_id = unk_id + 1 + (location_index)
                terminals.append(location_id)
                attn_success_cnt += 1
            else:
                attn_fail_cnt += 1
                terminals.append(unk_id)

    if json_dict['RightHandSide'] == 'BinaryCircuitExpression':
        terminals.append(terminal_dict['<empty>'])
        attn_que.append('<empty>')
        return process_binary_circuit(json_dict['RightHandSide'], terminals, attn_que, unk_id, attn_success_cnt, attn_fail_cnt)
    else:
        type = json_dict['Type']

        if type in terminal_dict.keys():
            terminals.append(terminal_dict[type])
            attn_que.append('Normal')
        else:
            if type in attn_que:
                location_index = [len(attn_que)-ind for ind,x in enumerate(attn_que) if x==type][-1]
                location_id = unk_id + 1 + (location_index)
                terminals.append(location_id)
                attn_success_cnt += 1
            else:
                attn_fail_cnt += 1
                terminals.append(unk_id)

    return (attn_success_cnt, attn_fail_cnt)

    

def get_typename(json_dict):
    typename = json_dict['Type']

    if typename is None:
        return

    if 'UnitExpression' in typename:
        return typename['UnitExpression'] + 'UnitExpression'
    elif 'DefaultType' in typename:
        return typename['DefaultType']['Value'] + 'DefaultType'

    return typename


def process(directory, terminal_dict, unk_id, attn_size, verbose=False, is_train=False):
    terminal_corpus = list()
    file_index = 0

    for file in os.listdir(directory):
        filename = os.path.join(directory, file)
        with open(filename) as f:
            print ('Start procesing %s !!!'%(filename))
            attn_que = deque(maxlen=attn_size)
            attn_success_total = 0
            attn_fail_total = 0
            length_total = 0
            file_index += 1

            data = json.loads(f.read())
            l = len(data)

            if l < 3e4:
                terminal_line = list()
                attn_que.clear()
                attn_success_cnt  = 0
                attn_fail_cnt  = 0

                items = []

                for section in data['Sections']:
                    items.append((section, False))
                    for entry in section['Entries']:
                        items.append((entry, False))
                        for param in entry['Parameters']:
                            items.append((param, True))

                for i, (json_dict, terminal) in enumerate(items):
                    if terminal:
                        dic_value = get_typename(json_dict)
                        if dic_value == 'BinaryCircuitExpression':
                            attn_success_cnt, attn_fail_cnt = process_binary_circuit(json_dict, terminal_line, attn_que, unk_id, attn_success_cnt, attn_fail_cnt)
                        elif dic_value in terminal_dict.keys():
                            terminal_line.append(terminal_dict[dic_value])
                            attn_que.append('Normal')
                        else:
                            if dic_value in attn_que:
                                location_index = [len(attn_que)-ind for ind,x in enumerate(attn_que) if x==dic_value][-1]
                                location_id = unk_id + 1 + (location_index)
                                terminal_line.append(location_id)
                                attn_success_cnt += 1
                            else:
                                attn_fail_cnt += 1
                                terminal.append(unk_id)
                    else:
                        terminal_line.append(terminal_dict['<empty>'])
                        attn_que.append('<empty>')

                terminal_corpus.append(terminal_line)
                attn_success_total += attn_success_cnt
                attn_fail_total += attn_fail_cnt
                #attn_total = attn_success_total + attn_fail_total
                #length_total = l

    return terminal_corpus


def save(filename, terminal_dict, terminal_num, vocab_size, attn_size, trainData, testData):
    with open(filename, 'wb') as f:
        save = {'terminal_dict': terminal_dict, 'terminal_num': terminal_num, 'vocab_size': vocab_size, 'attn_size': attn_size, 'trainData': trainData, 'testData': testData}
        pickle.dump(save, f, protocol=2)

if __name__ == '__main__':
    start_time = time.time()
    attn_size = 50
    terminal_dict_filename = './data/terminal_dict_TTI.pickle'
    target_filename = './data/TTI_terminal_whole.pickle'
    train_dir = "./train"
    test_dir = "./test"
    terminal_dict, terminal_num, vocab_size = restore_terminal_dict(terminal_dict_filename)
    trainData = process(train_dir, terminal_dict, vocab_size, attn_size, False, is_train=True)
    testData = process(test_dir, terminal_dict, vocab_size, attn_size, False, is_train=False)
    save(target_filename, terminal_dict, terminal_num, vocab_size, attn_size, trainData, testData)
    print('Finishing generating terminals and takes %.2f'%(time.time() - start_time))
