from encodings import utf_8
from glob import glob
import json
from random import sample
import shutil
import numpy as np
from typing import List
from collections import Counter
import sentencepiece as spm
import os
import pandas as pd
import csv
from itertools import repeat
import argparse

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class Datawriter:
    def __init__(self, batch_size, folder):
        self.batch_size = batch_size
        self.folder = folder
        self.total_count = 0
        self.current_file = 0
        self.cache = []

    def write(self, sample):
        self.cache.append(sample)
        if len(self.cache) >= self.batch_size:
            self.write_to_file()

    def write_to_file(self):
        with open(f"{self.folder}{self.current_file}.csv", "w") as file:
            for sample in self.cache:
                file.write(sample + "\n")
            self.current_file += 1
            self.total_count += len(self.cache)
            self.cache = []
        
        print(f"Wrote {self.total_count} samples to {self.current_file} files in {self.folder}.")


class NonTerminal:
    def __init__(self, name: str, children: list, optional=None, index=None, file=None):
        self.name = name
        self.children = children
        self.optional = optional
        self.index = index
        self.file = file

    def to_string(self):
        return self.name

    def to_dict(self):
        non_terminal = {
            "name": self.name,
            "children": [child.to_dict() for child in self.children]
        }

        if self.file:
            non_terminal["file"] = self.file

        return non_terminal

    def add_child(self, child):
        self.children.append(child)

    def tree(self):
        result = Node(self.name)
        [result.add_child(child.tree()) for child in self.children]

    @staticmethod
    def to_tree(json):
        for name, _ in json.items(): break
        children = [NonTerminal.to_tree(child) if "Parameters" in child else Terminal.to_tree(child) for child in json["Parameters"]]
        if "file" in json:
            return NonTerminal(name, children, json["file"])
        return NonTerminal(name, children)

class Terminal:
    def __init__(self, value, type, optional, index, generic_type=None):
        self.value = value
        self.type = type
        self.optional = optional
        self.index = index
        self.generic_type = generic_type

    def tree(self) -> None:
        return Node(self.to_string())

    def to_string(self):
        if self.generic_type is not None:
            return f"{self.generic_type.capitalize()}{self.type}"
        return f"{self.type}"

    def to_dict(self):
        return { "type":  self.type, "optional": self.optional, "index": self.index }

    @staticmethod
    def to_tree(json):
        return Terminal(json["type"], json["optional"], json["index"])


def dfs(visited, node):
    if node not in visited:
        visited.append(node)

    if hasattr(node, "children"):
        for child in node.children:
            dfs(visited, child)


def transform_ast(json_file):            
    def transform_ast_internal(json_dict):
        key = None
        value = None

        if len(json_dict) == 2:
            key, value = json_dict

        if key == "Sections":
            return NonTerminal(key, [transform_ast_internal(section) for section in value])
        elif value == "Entries":
            return NonTerminal(json_dict["Type"], [transform_ast_internal(entry) for entry in json_dict['Entries']])
        elif key == None:
            if json_dict['Type'] == "BinaryCircuitExpression":
                return NonTerminal("BinaryCircuitExpression", [transform_ast_internal(json_dict['LeftHandSide'])] + [transform_ast_internal(json_dict['RightHandSide'])])
            else:
                return Terminal(json_dict["Value"], json_dict["Type"], None, None)
        else: ## Entry
            children = []
            if not 'Parameters' in json_dict:
                return Terminal(json_dict['Value'], json_dict['Type'], None, None, None)

            for param in json_dict['Parameters']:
                type = param['Type']
                if type == 'BinaryCircuitExpression':
                    children.append(NonTerminal(type, [transform_ast_internal(param['LeftHandSide'])] + [transform_ast_internal(param['RightHandSide'])], param['Optional'], param['Index']))
                elif isinstance(type, dict) and type.get('DefaultType') is not None:
                    children.append(Terminal(type['DefaultType']['Value'], 'DefaultType', param['Optional'], param['Index'], type['DefaultType']['GenericType']))
                elif isinstance(type, dict) and type.get('UnitExpression') is not None:
                    children.append(Terminal(type['Value'], 'UnitExpression', param['Optional'], param['Index'], type['UnitExpression']))
                else:
                    type = param.get('Value') if param.get('Enum') == True else type
                    i = param['Index']
                    children.append(Terminal(param.get("Value", type), type, param['Optional'], i, param.get('Generic_Type')))

            return NonTerminal(json_dict['Type'], children)

    result = None

    with open(json_file, encoding="utf8") as f:
        for item in json.load(f).items():
            result = transform_ast_internal(item)
            break
    
    return result


def load_vocabulary(voc_file: str):
    voc = np.load(voc_file)
    voc_indexed = {}
    voc_inverse_indexed = {}

    for i, v in enumerate(voc):
        v = v.replace("Parameter", "")
        voc_indexed[v] = i
        voc_inverse_indexed[i] = v

    return voc_indexed, voc_inverse_indexed



def flatten_ast(ast) -> List[str]:
    visited = list()
    dfs(visited, ast)
    return [node.to_string() for node in visited]

def build_vocabulary_and_save(raw_asts_dir: str, counter_threshold: int = 2, vocab_for_bpe: bool = True, bpe_voc_size: int = 98):
    cnt = Counter()
    ast_count = 0

    for file in os.listdir(raw_asts_dir):
        ast_count += 1
        ast = transform_ast(os.path.join(raw_asts_dir, file))
        for token in flatten_ast(ast):
            if token is not None:
                cnt[token.split(':')[0]] += 1

    if vocab_for_bpe:
        voc_bpe_list = []
        for item, count in cnt.items():
            voc_bpe_list.extend([item] * count)

        with open('bpe_vocab.txt', 'w') as f:
            for item in list(voc_bpe_list):
                f.write(f"{item}\n")

        spm.SentencePieceTrainer.Train(input="bpe_vocab.txt", model_prefix='voc_bpe', vocab_size=bpe_voc_size, model_type='bpe', user_defined_symbols='</s>')
    else:
        voc = [item for item, count in cnt.items() if count >= counter_threshold]
        np.save("./data/voc.npy", np.array(voc))

        print("-- Vocabulary stats --")
        print(f"Amount of ASTs: {ast_count}")
        print(f"Threshold for vocabulary: {counter_threshold}")
        print(f"Vocabulary size: {len(voc)}")

        with open("vocab_unfiltered.txt", 'w') as f:
            for item, count in cnt.items():
                f.write(f"{item} {count}\n")

            f.flush()

def get_bpe_model(model_name: str):
    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(model_name)
    return sp_bpe



from bs4 import BeautifulSoup          

def extract(directory):
    def extract_internal(filename):
        try:
            with open(filename) as f:
                soup = BeautifulSoup(f.read(), "html.parser")
        except:
            with open(filename, encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

        rows = soup.find_all("tr")
    
        headers = {}
        thead = soup.find("thead")
        if thead:
            thead = thead.find_all("th")
            for i in range(len(thead)):
                headers[i] = thead[i].text.strip().lower()
        data = []
        for row in rows:
            cells = row.find_all("td")
            if thead:
                items = {}
                for index in headers:
                    text = cells[index].text
                    if text != "" and text != "..." and text != "-":
                        items[headers[index]] = text
            else:
                items = []
                for index in cells:
                    text = index.text
                    if text != "" and text != "..." and text != "...." and text != "-":
                        items.append(index.text.strip())
            data.append(items)
        return data

    d = dict()
    skip = ["keyword_tsa_cylinders.htm", 
            "keyword_tsa_equipmentsimple.htm", 
            "keyword_tsa_lossdetail.htm", 
            "keyword_tsa_operatingmode.htm", 
            "keyword_tsa_windingconnection.htm", 
            "keyword_tsa_windingdetail.htm", 
            "keyword_tsa_cores.htm",
            "keyword_tsa_general.htm",
            "keywordsoverview.htm"]

    for file in os.listdir(directory):
        if not file.endswith("htm") and not file.endswith("html") or file in skip: continue
        data = extract_internal(os.path.join(directory, file))
        key = data[0][0]
        innerdict = dict()

        for i in range(len(data)):
            n = len(data[i])
            if (i > 0 and n > 0):
                innerdict[data[i][0]] = data[i][1::]

        d[key] = innerdict
    
    with open("meta.json", "w") as outputfile:
        outputfile.write(json.dumps(d).replace("[", "{").replace("]", "}"))


def get_type(token):
    split_token = token.split(":")

    if len(split_token) > 1:
        return split_token[1]

    return None


def raw_asts_to_samples_save(directory, vocab_file, lookback_tokens = 100):
    vocab, vocab_inverse = load_vocabulary(vocab_file)
    out_of_voc = len(vocab)
    unknown_token = out_of_voc
    padding_token = out_of_voc + 1
    ast_count = 0
    sample_count = 0
    ast_no_calls = 0
    correct_types = 0
    incorrect_types_unfixed = 0
    incorrect_types_fixed = 0
    data_writer = Datawriter(100, ".")

    for file in os.listdir(directory):
        ast_count += 1
        ast = transform_ast(os.path.join(directory, file))
        ast_flat_encoding = []
        entries = []
        sections = []

        for i, token in enumerate(flatten_ast(ast)):
            if token in vocab and token.lower().endswith("call"):
                entries.append(i)
                ast_flat_encoding.append(vocab[token])
            elif token in vocab and token.lower().endswith("section"):
                sections.append(i)
                ast_flat_encoding.append(vocab[token])
            elif token in vocab:
                ast_flat_encoding.append(vocab[token])
            else:
                ast_flat_encoding.append(unknown_token)

        #if len(call_locations) == 0:
        #    ast_no_calls += 1
        l_entries = len(entries)

        for i, section in enumerate(sections):
            sample_encoding = list(repeat(padding_token, lookback_tokens - 1))
            sample_encoding.insert(0, ast_flat_encoding[section])
            sample_encoding.insert(0, 1)
            sample_encoding.append(ast_flat_encoding[section + 1])

        for i, section in enumerate(sections):
            if i < len(sections) - 1:
                sample_encoding = ast_flat_encoding[section:sections[i+1]]
                label = ast_flat_encoding[sections[i + 1]]
            else:
                sample_encoding = ast_flat_encoding[section:entries[l_entries - 1]]
                label = ast_flat_encoding[entries[l_entries - 1]]
            
            if len(sample_encoding) >= lookback_tokens:
                sample_encoding = sample_encoding[:lookback_tokens-1]

            while len(sample_encoding) < lookback_tokens:
                sample_encoding.append(padding_token)

            sample_encoding.insert(0, len([encoding for encoding in sample_encoding if encoding != padding_token]))
            sample_encoding.append(label)

            #data_writer.write(','.join(['%.5f' % num for num in sample_encoding]))
            sample_count += 1
    #data_writer.write_to_file()

        section = ast_flat_encoding[0]

        for i, entry in enumerate(entries):
            if vocab_inverse[ast_flat_encoding[entry - 1]].lower().endswith("section"):
                if i < len(entries) - 1:
                    sample_encoding = ast_flat_encoding[entry-1:entries[i+1]]
                else:
                    sample_encoding = ast_flat_encoding[entry:]
                section = entry - 1
            else:
                sample_encoding = ast_flat_encoding[section:entry]

            if i < len(entries) - 1:
                e = ast_flat_encoding[entry:entries[i+1]]
            else:
                e = ast_flat_encoding[entry:]

            if len(sample_encoding) >= lookback_tokens:
                sample_encoding = sample_encoding[:lookback_tokens-1]

            while len(sample_encoding) < lookback_tokens:
                sample_encoding.append(padding_token)

            sample_encoding.insert(0, len([encoding for encoding in sample_encoding if encoding != padding_token]))
            sample_encoding.append(ast_flat_encoding[entry])
            sample_count += 1

            j = 0
            l = len(e) - 1
            while j < l:
                sample_encoding = e[:j+1]
                sample_encoding.insert(0, len(sample_encoding))
                sample_encoding.extend(list(repeat(padding_token, lookback_tokens - 1 - j)))
                sample_encoding.append(e[j + 1])
                j += 1


    print("-- Training encoding stats --")
    print(f"Amount of ASTs: {ast_count}")
    print(f"Amount of ASTs without method calls: {ast_no_calls}")
    print(f"Ratio incorrect types: {(incorrect_types_unfixed + incorrect_types_fixed)} / {(incorrect_types_unfixed + incorrect_types_fixed + correct_types)}")
    print(f"Ratio fixed types: {(incorrect_types_fixed)} / {incorrect_types_fixed + incorrect_types_unfixed}")
    print(
        f"Total training samples: {sample_count} from"
        f" {ast_count- ast_no_calls} ASTs"
    )

parser = argparse.ArgumentParser(description="Arguments for ast utils",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--directory", help="directory to save samples")
args = parser.parse_args()
# build_vocabulary_and_save(directory, vocab_for_bpe=False)
raw_asts_to_samples_save(args.directory, './data/voc.npy')