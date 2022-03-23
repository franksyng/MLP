import pandas as pd
import os
import re

import torch
from transformers import *
import transformers
import torch.nn as nn
import sentencepiece

# def split_chunks(sentence, chunk_num):
#     output = []
#     for i in range(0, len(sentence), chunk_num):
#         output.append(sentence[i:i + chunk_num])
#     return output
#
#
# def obtain_mini_patch(train_data, VOCAB, length):
#     output = []
#     for note in train_data:
#         sentence = note[0]
#         label = note[1]
#         sentence = split_chunks(sentence, length)
#         label = split_chunks(label, length)
#         for i in range(len(sentence)):
#             curr_sentence = sentence[i]
#             curr_label = label[i]
#             curr_sentence = ['[CLS]'] + curr_sentence
#             curr_label = [VOCAB['[CLS]']] + curr_label
#             curr_len = len(curr_sentence) - 1
#             while curr_len < length:
#                 curr_sentence.append('[PAD]')
#                 curr_label.append(VOCAB['[PAD]'])
#                 curr_len += 1
#             curr_sentence.append('[SEP]')
#             curr_label.append(VOCAB['[SEP]'])
#             output.append((curr_sentence, curr_label))
#     return output


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
        rule = re.compile('[\n\r\t,()\"\'.]')
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
            for i in range(len(labels)):
                if labels[i] == 'O':
                    continue
                elif i == 0:
                    labels[i] = 'B-' + str(labels[i])
                else:
                    former_tag = labels[i - 1].split('-')
                    if len(former_tag) == 1:
                        former_tag = former_tag[0]
                    else:
                        former_tag = former_tag[1]
                    if labels[i] == former_tag:
                        labels[i] = 'I-' + str(labels[i])
                    else:
                        labels[i] = 'B-' + str(labels[i])

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

    def read_label(self):
        df_feature = pd.read_csv(self.feature_data, index_col=False)
        df_feature['idx'] = df_feature.index
        df_feature = df_feature.set_index('feature_num')
        labels = df_feature.to_dict()['idx']
        idx = 0
        vocab = {}
        for label in labels:
            b_tag = 'B-' + str(labels[label])
            i_tag = 'I-' + str(labels[label])
            vocab[idx] = b_tag
            vocab[idx + 1] = i_tag
            idx += 2
        vocab['O'] = 286
        vocab['[CLS]'] = 287
        vocab['[SEP]'] = 288
        vocab['[PAD]'] = 289
        return vocab

    def tag_to_idx(self, label, vocab):
        for i in range(len(label)):
            label[i] = vocab[label[i]]
        return label

    # def make_dataset(self):
    #     ready_data = self.loader()
    #     tokenized = []
    #     tokenizer = AutoTokenizer.from_pretrained('jsylee/scibert_scivocab_uncased-finetuned-ner')
    #     start = tokenizer('[CLS]', add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    #     end = tokenizer('[SEP]', add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    #     vocab = self.read_label()
    #     for each_set in ready_data:
    #         curr_sentence = torch.empty(0)
    #         curr_label = []
    #         sentence = each_set[0]
    #         label = each_set[1]
    #         for i in range(len(sentence)):
    #             wordpiece = tokenizer(sentence[i], add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    #             curr_sentence = torch.cat((curr_sentence, wordpiece), dim=0)
    #             if len(wordpiece) == 1:
    #                 curr_label += [label[i]]
    #             else:
    #                 curr_label += [label[i]] * len(wordpiece)
    #         curr_sentence = torch.cat((start, curr_sentence), dim=0)
    #         curr_sentence = torch.cat((curr_sentence, end), dim=0)
    #         curr_sentence = curr_sentence.type(torch.int64)
    #         curr_label.insert(0, '[CLS]')
    #         curr_label.append('[SEP]')
    #         curr_label = self.tag_to_idx(curr_label, vocab)
    #         tokenized.append((curr_sentence, curr_label))
    #     return tokenized
    # def make_dataset(self):
    #     ready_data = self.loader()
    #     vocab = self.read_label()

# data_path = 'data/train.csv'
# pn_path = 'data/patient_notes.csv'
# feature_path = 'data/features.csv'
# preprocessor = Preprocessor(data_path, pn_path, feature_path)
# # dataset = preprocessor.to_dataframe()
# print(len(preprocessor.read_label()))
