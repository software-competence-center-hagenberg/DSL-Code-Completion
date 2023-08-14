import numpy as np
from six.moves import cPickle as pickle
import json
from collections import Counter
import time
import os

EMPTY_TOKEN = '<empty>'
freq_dict = Counter()
terminal_num = set()
terminal_num.add(EMPTY_TOKEN)

def process_binary_circuit(json_dict, terminals):
    if json_dict['LeftHandSide'] == 'BinaryCircuitExpression':
        freq_dict[EMPTY_TOKEN] += 1
        process_binary_circuit(json_dict['LeftHandSide'], terminals)
    else:
        terminals.add(json_dict['Type'])
        freq_dict[json_dict['Type']] += 1

    if json_dict['RightHandSide'] == 'BinaryCircuitExpression':
        freq_dict[EMPTY_TOKEN] += 1
        process_binary_circuit(json_dict['RightHandSide'], terminals)
    else:
        terminals.add(json_dict['Type'])
        freq_dict[json_dict['Type']] += 1


def get_typename(json_dict):
    typename = json_dict['Type']

    if typename is None:
        return

    if 'UnitExpression' in typename:
        return typename['UnitExpression'] + 'UnitExpression'
    elif 'DefaultType' in typename:
        return typename['DefaultType']['Value'] + 'DefaultType'

    return typename


def process(directory):
    for file in os.listdir(directory):
        filename = os.path.join(directory, file)
        with open(filename) as f:
            print ('Start procesing %s !!!'%(filename))

            data = json.loads(f.read())
            if len(data) < 3e4:

                items = []

                for section in data['Sections']:
                    items.append((section, False))
                    for entry in section['Entries']:
                        items.append((entry, False))
                        for param in entry['Parameters']:
                            items.append((param, True))

                for i, (json_dict, terminal) in enumerate(items):
                    if terminal:
                        value = get_typename(json_dict)
                        if value == 'BinaryCircuitExpression':
                            process_binary_circuit(json_dict, terminal_num)
                        else:
                            terminal_num.add(value)
                            freq_dict[value] += 1
                    else:
                        freq_dict[EMPTY_TOKEN] += 1


def save(filename):
    with open(filename, 'wb') as f:
        save = {'freq_dict': freq_dict,'terminal_num': terminal_num}
        pickle.dump(save, f, protocol=2)


if __name__ == '__main__':
    train_dir = "./train"
    test_dir = "./test"
    target_filename = './data/freq_dict_TTI.pickle'
    start_time = time.time()
    process(train_dir)
    process(test_dir)
    save(target_filename)
    print(freq_dict['EmptY'], freq_dict['Empty'], freq_dict['empty'], freq_dict['EMPTY'], freq_dict[EMPTY_TOKEN])
    print('Finishing generating freq_dict and takes %.2f'%(time.time() - start_time))