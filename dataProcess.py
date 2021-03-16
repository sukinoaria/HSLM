from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import  numpy as np
import scipy.sparse as sp

import torch

sys.path.append("../")
from basic.alphabet import Alphabet
from basic.data import InputFeatures

word_alphabet = Alphabet('word', True)

asp_label_alphabet = Alphabet('asp label', True)
asp_label_alphabet.add("O")
asp_label_alphabet.add("B")
asp_label_alphabet.add("I")

opi_label_alphabet = Alphabet('opi label', True)
opi_label_alphabet.add("O")
opi_label_alphabet.add("B")
opi_label_alphabet.add("I")

polarity_alphabet = Alphabet('relation', True)
polarity_alphabet.add("NEG")
polarity_alphabet.add("NEU")
polarity_alphabet.add("POS")

char_alphabet = Alphabet('char', True)

def readDataFromFile(path):
    f = open(path, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    seq = 0
    datasets = []
    words = []
    labels = []
    relations = []
    for l in lines:
        if l.strip() == "#Relations":
            continue
        elif l.strip() == "" and len(words)>0:
            datasets.append({"words": words, "labels": labels, "relations": relations})
            if len(words)>seq:
                seq = len(words)
            words = []
            labels = []
            relations = []
        elif len(l.strip().split("\t")) == 2:
            tempLine = l.strip().split("\t")
            # WORD
            words.append(tempLine[0].lower())
            # LABEL
            labels.append(tempLine[1])
        #gold relation 
        elif len(l.strip().split("\t")) == 5:
            rel = l.strip().split("\t")
            rel[:4] = list(map(int, rel[0: 4]))
            relations.append(rel)
    print("max_seq_length:"+str(seq))
    return datasets

def convert_examples_to_features(examples, max_seq_length=100, max_char_length=10):
    seq = 0
    charSeq = 0
    features = []
    num = 0
    char_num = 0
    relationlen = 0
    sentlen = 0
    for (example_index, example) in enumerate(examples):

        tokens = []
        token_ids = []
        token_mask = []
        chars = []
        char_ids = []
        char_mask = []
        charLength = []
        tokenLength = []
        asp_labels = []
        asp_label_ids = []
        opi_labels = []

        #### split words and labels ####
        for (i, token) in enumerate(example["words"]):
            char = []
            charId = []
            tokens.append(token)
            word_alphabet.add(token)
            token_ids.append(word_alphabet.get_index(token))
            for w in token:
                char.append(w)
                char_alphabet.add(w)
                charId.append(char_alphabet.get_index(w))
            char_ids.append(charId)
            chars.append(char)
            label = example["labels"][i]
            if label == "B-T" or label == "I-T":
                asp_labels.append(label[0])
                asp_label_ids.append(asp_label_alphabet.get_index(label[0]))
            else:
                asp_labels.append("O")
                asp_label_ids.append(asp_label_alphabet.get_index("O"))

        #### update polaritys and aspect mapped opinion label ####
        gold_relations = example["relations"]

        # tag mapping O:1 B:2 I:3  padding:0
        opi_label_ids = np.zeros((max_seq_length,max_seq_length))
        opi_label_ids[:len(tokens),:len(tokens)] = 1
        polaritys = np.zeros(max_seq_length)
        for gr in gold_relations:
            polarity_alphabet.add(gr[4])
            polaritys[gr[2]:gr[3]] = polarity_alphabet.get_index(gr[4])
            opi_label_ids[gr[2]:gr[3], gr[0]] = opi_label_alphabet.get_index('B')
            opi_label_ids[gr[2]:gr[3], gr[0]+1:gr[1]] = opi_label_alphabet.get_index('I')

        # sparse opi matrix for save
        opi_label_ids = sp.csr_matrix(opi_label_ids)
        if len(tokens)>seq:
            seq = len(tokens)
        if len(tokens)>max_seq_length:
            num+=1
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length:
            print("error occured..")
            exit(0)
        for i in range(len(char_ids)):
            if len(char_ids[i])>charSeq:
                charSeq = len(char_ids[i])
            if len(char_ids[i])>max_char_length:
                char_num +=1
            # if len(char_ids[i]) > max_char_length:
                char_ids[i] = char_ids[i][:max_char_length]
            charLength.append(len(char_ids[i]))

            char_m = [1]*len(char_ids[i])
            while len(char_ids[i])<max_char_length:
                char_ids[i].append(0)
                char_m.append(0)
            char_mask.append(char_m)
        #### make token_ids ####
        token_mask = [1] * len(tokens)
        tokenLength.append(len(tokens))
        while len(token_ids) < max_seq_length:
            token_ids.append(0)
            asp_label_ids.append(0)
            token_mask.append(0)
            char_mask.append([0]*max_char_length)
            char_ids.append([0]*max_char_length)
            charLength.append(0)

        sentlen+=1
        relationlen+=len(gold_relations)
        features.append(
            InputFeatures(
                tokens,
                token_ids,
                token_mask,
                chars,
                char_ids,
                char_mask,
                charLength,
                tokenLength,
                asp_labels,
                opi_labels,
                asp_label_ids,
                opi_label_ids,
                polaritys,
                gold_relations))
    print(seq)
    print(num)
    print("char length:")
    print(charSeq)
    print(char_num)
    print(sentlen)
    print(relationlen)

    print("\n")
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_file", default="text_data/14lap/train_14lap.txt", type=str)
    parser.add_argument("--dev_file", default="text_data/14lap/dev_14lap.txt",type=str)
    parser.add_argument("--test_file", default="text_data/14lap/test_14lap.txt",type=str)

    parser.add_argument("--output_file", default="bin_data/14lap",type=str)
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=150, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()

    if not os.path.exists("bin_data"):
        os.mkdir("bin_data")

    train_set = readDataFromFile(args.train_file)
    dev_set = readDataFromFile(args.dev_file)
    test_set = readDataFromFile(args.test_file)

    train_features = convert_examples_to_features(train_set ,max_seq_length=83, max_char_length=20)
    dev_features = convert_examples_to_features(dev_set , max_seq_length=69, max_char_length=20)
    test_features = convert_examples_to_features(test_set , max_seq_length=71, max_char_length=20)
    print(word_alphabet.size())

    torch.save({"train":train_features,"dev":dev_features,"test":test_features,"word_alpha":word_alphabet,"asp_alpha":
                asp_label_alphabet,"opi_alpha":opi_label_alphabet,"polar_alpha":polarity_alphabet,"char_alpha":char_alphabet},args.output_file)
