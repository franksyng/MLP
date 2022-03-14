import pandas as pd
import os
import re


def read_label(path):
    feature = os.path.join(path, 'features.csv')
    df_feature = pd.read_csv(feature, index_col=False)
    df_feature = df_feature.set_index('feature_num')
    df_feature['feature_num'] = df_feature.index
    labels = df_feature.to_dict()['feature_num']
    labels['CLS'] = 97
    labels['SEP'] = 98
    labels['PAD'] = 99
    return labels


def split_chunks(sentence, chunk_num):
    output = []
    for i in range(0, len(sentence), chunk_num):
        output.append(sentence[i:i + chunk_num])
    return output


def obtain_mini_patch(train_data, VOCAB, length):
    output = []
    for note in train_data:
        sentence = note[0]
        label = note[1]
        sentence = split_chunks(sentence, length)
        label = split_chunks(label, length)
        for i in range(len(sentence)):
            curr_sentence = sentence[i]
            curr_label = label[i]
            curr_sentence = ['CLS'] + curr_sentence
            curr_label = [VOCAB['CLS']] + curr_label
            curr_len = len(curr_sentence) - 1
            while curr_len < length:
                curr_sentence.append('PAD')
                curr_label.append(VOCAB['PAD'])
            curr_sentence.append('SEP')
            curr_label.append(VOCAB['SEP'])
            output.append((curr_sentence, curr_label))
    return output


class DataProvider:
    def __init__(self, path):
        self.train_data = os.path.join(path, 'train.csv')
        self.pn_data = os.path.join(path, 'patient_notes.csv')

    def preprocess_data(self):
        """
        Remove unnecessary chars and setup link between index and labels for each location annotation
        Annotation here could be a single list [], or multi-dimension list [[], [], []]
        """
        df_train = pd.read_csv(self.train_data, index_col=False)  # Load data
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
                            labels.append(loc[1])
                else:
                    train.append(mid)
                    labels.append(loc[1])
                pn = left
            if len(pn) != 0:
                if ' ' in pn:
                    for each in pn.split(' ')[::-1]:
                        if len(each) != 0:
                            train.append(each)
                            labels.append('O')

            train = train[::-1]
            labels = labels[::-1]
            dataset.append((train, labels))
        return dataset

path = 'data'
provider = DataProvider(path)
train = provider.loader()
tag_to_idx = read_label(path)
train_data = obtain_mini_patch(train, tag_to_idx, 10)
