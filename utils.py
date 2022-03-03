import pandas as pd
import os
import re


def data_preprocessing(data):
    # Set path
    raw_train_data = os.path.join(data, 'train.csv')
    pn_data = os.path.join(data, 'patient_notes.csv')

    # Load data
    df_pn = pd.read_csv(pn_data, index_col=False)
    df_train = pd.read_csv(raw_train_data, index_col=False)
    df_train = df_train[['pn_num', 'feature_num', 'location']]

    rule = re.compile('[\d\s;,]')
    merged_train_data = df_train.merge(df_pn, how='inner', on='pn_num').reset_index()
    merged_train_data = merged_train_data[merged_train_data.location != '[]']  # Exclude empty data
    preprocessed_data = merged_train_data.to_dict()
    temp_loc_dict = merged_train_data['location'].to_dict()
    # print(temp_loc_dict)

    for keys in temp_loc_dict:
        loc = []
        valid_loc = rule.findall(temp_loc_dict[keys])
        valid_loc = ''.join(valid_loc).split(',')
        for loc_idx in valid_loc:
            if ';' not in loc_idx:
                loc_idx = loc_idx.strip().split(' ')
                loc.append([int(loc_idx[0]), int(loc_idx[1])])
                # print(loc_idx)
            elif ';' in loc_idx:
                loc_idx = loc_idx.strip().split(';')
                # print(loc_idx)
                for sub_idx in loc_idx:
                    sub_idx = sub_idx.split(' ')
                    loc.append([int(sub_idx[0]), int(sub_idx[1])])
                    # print(loc_idx)
        preprocessed_data['location'][keys] = loc
    return preprocessed_data


def data_provider(data):
    text = []
    label = []
    output_set = []
    for idx in data['index']:
        # curr_text = []
        # curr_label = []
        loc_list = data['location'][idx]
        for loc in loc_list:
            text.append(data['pn_history'][idx][loc[0]:loc[1]])
            label.append(data['feature_num'][idx])
    output_set.append((text, label))
    return output_set


# def data_to_idx(training_data):
#     start_tag = "<START>"
#     stop_tag = "<STOP>"
#     word_to_idx = {}
#     label_to_idx = {}
#     label_idx = 0
#     for texts, labels in training_data:
#         for phrase in texts:
#             word_to_idx[phrase] = len(word_to_idx)
#         for label in labels:
#             if label not in label_to_idx:
#                 label_to_idx[label] = label_idx
#                 label_idx += 1
#     label_to_idx[start_tag] = label_idx
#     label_to_idx[stop_tag] = label_idx + 1
#     return word_to_idx, label_to_idx


