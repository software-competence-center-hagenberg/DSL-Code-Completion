import numpy as np
from six.moves import cPickle as pickle
import json
import time
from collections import Counter, defaultdict
import os
from pandas.io.json import json_normalize
from flatten_json import flatten
import argparse

typeDict = dict()
numID = set()
no_empty_set = set()
typeList = list()
numType = 0
dicID = dict()    

    
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
    corpus_N = list()
    corpus_parent = list()

    for file in os.listdir(directory):
        filename = os.path.join(directory, file)
        with open(filename) as f:
            print ('Start procesing %s !!!'%(filename))
            line_index = 0

            line_index += 1
            if line_index % 1000 == 0:
                print('Processing line:', line_index)
            data = json.loads(f.read())
            line_N = list()
            has_sibling = Counter()
            parent_counter = defaultdict(lambda: 1)
            parent_list = list()
            k = 0

            if len(data) >= 3e4:
                continue

            items = []

            for section in data['Sections']:
                items.append(section)
                for entry in section['Entries']:
                    items.append(entry)
                    for param in entry['Parameters']:
                        items.append(param)

            for i, dict in enumerate(items):                
                typeName = get_typename(dict)

                if typeName is None:
                    continue

                if typeName in typeList:
                    base_ID = typeDict[typeName]
                else:
                    typeList.append(typeName)
                    global numType
                    typeDict[typeName] = numType
                    base_ID = numType
                    numType = numType + 1

                entries = 'Entries' in dict.keys()
                params = 'Parameters' in dict.keys()
                if entries or params:
                    if has_sibling[i]:
                        ID = base_ID * 4 + 3
                    else:
                        ID = base_ID * 4 + 2

                    childs = dict['Entries'] if entries else dict['Parameters']
                    l = len(childs)
                    for j in range(l):
                        j = j + k
                        parent_counter[j] = j - i
                    
                    if l > 1:
                        for j in range(l):
                            j = j + k
                            has_sibling[j] = 1
                else:
                    if has_sibling[i]:
                        ID = base_ID * 4 + 1
                    else:
                        ID = base_ID * 4

                if 'Value' in dict.keys():
                    no_empty_set.add(ID)

                line_N.append(ID)
                parent_list.append(parent_counter[i])
                numID.add(ID)
                k += l


            corpus_N.append(line_N)
            corpus_parent.append(parent_list)
    
    return corpus_N, corpus_parent


def map_dense_id(data):
    result = list()
    for line_id in data:
        line_new_id = list()
        for i in line_id:
            if i in dicID.keys():
                line_new_id.append(dicID[i])
            else:
                dicID[i] = len(dicID)
                line_new_id.append(dicID[i])
        result.append(line_new_id)
    return result


def save(filename, typeDict, numType, dicID, vocab_size, trainData, testData, trainParent, testParent, empty_set_dense):
    with open(filename, 'wb') as f:
        save = {'vocab_size': vocab_size, 'trainData': trainData, 'testData': testData, 'trainParent': trainParent, 'testParent': testParent}
        pickle.dump(save, f, protocol=2)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Arguments for get non terminal", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dir", help="directory of training files")
    parser.add_argument("--test_dir", help="directory of test files")
    args = parser.parse_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    target_filename = "./data/TTI_non_terminal.pickle"
    trainData, trainParent = process(train_dir)
    testData, testParent = process(test_dir)
    trainData = map_dense_id(trainData)
    testData = map_dense_id(testData)
    vocab_size = len(numID)
    empty_set = numID.difference(no_empty_set)
    empty_set_dense = set()

    for i in empty_set:
        empty_set_dense.add(dicID[i])

    print('The N set that can only has empty terminals: ',len(empty_set_dense), empty_set_dense)
    print('The vocaburary:', vocab_size, numID)
    
    save(target_filename, typeDict, numType, dicID, vocab_size, trainData, testData, trainParent, testParent,empty_set_dense)
    print('Finishing generating terminals and takes %.2fs'%(time.time() - start_time))