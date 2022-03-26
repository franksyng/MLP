import pandas as pd
import os
import re

import torch
from transformers import *
import transformers
import torch.nn as nn
import sentencepiece
import numpy as np
from sklearn.metrics import classification_report


def cal_accuracy(preds, label_ids, mask):
    valid_len = np.sum(mask)
    flat_preds = preds.to('cpu').numpy().flatten()[:valid_len]
    flat_labels = label_ids.flatten()[:valid_len]
    acc = classification_report(flat_labels, flat_preds, output_dict=True)['accuracy']
    new_labels = [i for i in flat_labels if i != 286]
    new_labels = list(dict.fromkeys(new_labels))
    target_names = [str(i) for i in new_labels]
    ner_f1 = classification_report(flat_labels, flat_preds, labels=new_labels, target_names=target_names, output_dict=True)
    return acc, ner_f1['macro avg']['f1-score']


def char_embedding(data_loader):
    tkn_list = []
    for _, data in enumerate(data_loader, 0):
        tkn_result = data['tkn_result']
        para_list = []
        for para in tkn_result:
            for word in para:
                para_list.append(char_to_idx(word))
        tkn_list.append(para_list)
    return tkn_list


def char_to_idx(word):
    pad = [0] * 10
    if word == '[CLS]' or word == '[SEP]' or word == '[PAD]':
        return pad
    else:
        vocab = char_vocab()
        chars = [vocab[char] for char in word]
        chars.extend(pad)
        return chars[:10]


def char_vocab():
    vocab = {}
    counter = 1
    for i in range(91, 127):
        vocab[chr(i)] = counter
        counter += 1
    for j in range(31, 65):
        vocab[chr(j)] = counter
        counter += 1
    return vocab


def count_label(labels):
    stat = {}
    for para in labels:
        for word_idx in para:
            if word_idx in stat.keys():
                stat[word_idx] += 1
            else:
                stat[word_idx] = 1
    return stat


def split_dataset(sentences, labels, train_percent, val_percent, test=None):
    sentences = [''.join([s[0] for s in sent]) for sent in sentences]
    # print(sentences)
    train_size = int(int(train_percent * len(sentences)) / 10)
    print('sssss', train_size)
    val_size = int(int(val_percent * len(sentences)) / 10)
    print('aaaaa', val_size)
    train_sentence = []
    train_label = []
    val_sentence = []
    val_label = []
    patient_case_idx = 100
    while True:
        if patient_case_idx > len(sentences):
            break
        if patient_case_idx == 100:
            sen_case = sentences[0:patient_case_idx]
            train_sentence.extend(sen_case[0:train_size])
            val_sentence.extend(sen_case[train_size:])
            lab_case = labels[0:patient_case_idx]
            train_label.extend(lab_case[0:train_size])
            val_label.extend(lab_case[train_size:])
            patient_case_idx += 100
        else:
            sen_case = sentences[patient_case_idx - 100:patient_case_idx]
            train_sentence.extend(sen_case[0:train_size])
            val_sentence.extend(sen_case[train_size:])
            lab_case = labels[patient_case_idx - 100:patient_case_idx]
            train_label.extend(lab_case[0:train_size])
            val_label.extend(lab_case[train_size:])
            patient_case_idx += 100

    return (train_sentence, train_label), (val_sentence, val_label)


def sentence_and_label(getter_sentences, tag2dix):
    tag2idx = tag2dix
    sentences = [' '.join([s[0] for s in sent]) for sent in getter_sentences]
    labels = [[s[1] for s in sent] for sent in getter_sentences]
    labels = [[tag2idx.get(l) for l in lab] for lab in labels]
    return sentences, labels


class Preprocessor:

    def __init__(self, data_path, pn_path, feature_path):
        self.data = data_path
        self.pn_data = pn_path
        self.feature_data = feature_path

    def preprocess_data(self):
        """
        Remove unnecessary chars and setup link between index and labels for each location annotation
        Annotation here could be a single list [], or multi-dimension list [[], [], []]
        """
        df_train = pd.read_csv(self.data, index_col=False)  # Load data
        df_train = df_train[['pn_num', 'feature_num', 'location']]
        rule = re.compile('[\d\s;,]')  # Exclude not needed chars
        df_train = df_train[df_train.location != '[]']  # Exclude empty data
        df_train = df_train.groupby('pn_num').agg({'feature_num': lambda x: list(x), 'location': lambda x: list(x)})  # Aggregate locations
        preprocessed = df_train.to_dict()
        temp_data = df_train.to_dict()['location']

        for keys in temp_data:
            loc = []
            for each_label in temp_data[keys]:
                sub_loc = []
                valid_loc = rule.findall(each_label)
                valid_loc = ''.join(valid_loc).split(',')  # Split to list
                # Deal with ';' in location labels
                for loc_idx in valid_loc:
                    if ';' not in loc_idx:
                        loc_idx = loc_idx.strip().split(' ')
                        sub_loc.append([int(loc_idx[0]), int(loc_idx[1])])
                    elif ';' in loc_idx:
                        loc_idx = loc_idx.strip().split(';')
                        for sub_idx in loc_idx:
                            sub_idx = sub_idx.split(' ')
                            sub_loc.append([int(sub_idx[0]), int(sub_idx[1])])
                loc.append(sub_loc)
            preprocessed['location'][keys] = loc
        return preprocessed

    def sort_annotation(self):
        """
        Sort location list in format of each single label (e.g. [[5, 8], 200], where indicates index 5-8 and label is 200)
        """
        preprocessed = self.preprocess_data()
        train_data = {}
        for key in preprocessed['location']:
            loc = preprocessed['location'][key]
            label = preprocessed['feature_num'][key]
            loc_with_label = []
            for i in range(len(label)):
                if len(loc[i]) != 1:
                    for sub_loc in loc[i]:
                        loc_with_label.append([sub_loc, label[i]])
                else:
                    loc_with_label.append([loc[i][0], label[i]])
            loc_with_label = sorted(loc_with_label, key=lambda x: x[0][0], reverse=True)
            train_data[key] = loc_with_label
        return train_data

    def loader(self):
        """
        Use the sorted location list to segment training data.
        Words and annotations are linked with list indices.
        e.g. ['Frank', 'is', 'a', 'handsome', 'guy']
             ['LABEL', 'O', 'O', 'LABEL', 'O']
             where 'O' indicates no label, and LABEL in real cases are feature_num
        Training set be like:
        train = [([words], [labels]), ([words], [labels])...]
        """
        train_data = self.sort_annotation()
        df_pn = pd.read_csv(self.pn_data, index_col=False)
        df_pn = df_pn.set_index('pn_num')
        pn_data = df_pn.to_dict()
        rule = re.compile('[\n\r\t,()\"\'.\-:;/]')
        dataset = []
        for key in train_data:
            train = []
            labels = []
            annotations = train_data[key]
            pn = pn_data['pn_history'][key].lower()
            pn = rule.sub(' ', pn)
            for loc, entity in annotations:
                left = pn[:loc[0]]
                mid = pn[loc[0]:loc[1]]
                right = pn[loc[1]:]
                if ' ' in right:
                    for each in right.split(' ')[::-1]:
                        if len(each) != 0:
                            train.append(each)
                            labels.append('O')
                else:
                    train.append(right)
                    labels.append('O')
                if ' ' in mid:
                    for each in mid.split(' ')[::-1]:
                        if len(each) != 0:
                            train.append(each)
                            labels.append(str(entity))
                else:
                    train.append(mid)
                    labels.append(str(entity))
                pn = left
            if len(pn) != 0:
                if ' ' in pn:
                    for each in pn.split(' ')[::-1]:
                        if len(each) != 0:
                            train.append(each)
                            labels.append('O')

            train = train[::-1]
            labels = labels[::-1]
            # for i in range(len(labels)):
            #     if labels[i] == 'O':
            #         continue
            #     elif i == 0:
            #         labels[i] = 'B-' + str(labels[i])
            #     else:
            #         former_tag = labels[i - 1].split('-')
            #         if len(former_tag) == 1:
            #             former_tag = former_tag[0]
            #         else:
            #             former_tag = former_tag[1]
            #         if labels[i] == former_tag:
            #             labels[i] = 'I-' + str(labels[i])
            #         else:
            #             labels[i] = 'B-' + str(labels[i])

            dataset.append((train, labels))
        return dataset

    def to_dataframe(self):
        data_with_tag = self.loader()
        dataset = []
        counter = 0
        for note in data_with_tag:
            for i in range(len(note[0])):
                curr_row = [counter, note[0][i], note[1][i]]
                dataset.append(curr_row)
            counter += 1
        dataset = pd.DataFrame(dataset, columns=['sentence_idx', 'word', 'tag'])
        return dataset

    def make_vocab_iob(self):
        df_feature = pd.read_csv(self.feature_data, index_col=False)
        df_feature['idx'] = df_feature.index
        df_feature = df_feature.set_index('feature_num')
        labels = df_feature.to_dict()['idx']
        idx = 0
        vocab = {}
        for label in labels:
            b_tag = 'B-' + str(label)
            i_tag = 'I-' + str(label)
            vocab[b_tag] = idx
            vocab[i_tag] = idx + 1
            idx += 2
        vocab['O'] = 286
        return vocab

    def make_vocab(self):
        df_feature = pd.read_csv(self.feature_data, index_col=False)
        df_feature['idx'] = df_feature.index
        df_feature = df_feature.set_index('feature_num')
        labels = df_feature.to_dict()['idx']
        idx = 0
        vocab = {}
        for label in labels:
            vocab[str(label)] = idx
            idx += 1
        vocab['O'] = 143
        return vocab


# data_path = 'data/train.csv'
# pn_path = 'data/patient_notes.csv'
# feature_path = 'data/features.csv'
# preprocessor = Preprocessor(data_path, pn_path, feature_path)
# dataset = preprocessor.to_dataframe()
# print(preprocessor.make_vocab())
