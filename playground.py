import pandas as pd
import os
import re

root = ''
train_path = os.path.join(root, 'data/train.csv')
pn_path = os.path.join(root, 'data/patient_notes.csv')
feature_path = os.path.join(root, 'data/features.csv')

df_pn = pd.read_csv(pn_path, index_col=False)
df_train = pd.read_csv(train_path, index_col=False)
df_train = df_train[['pn_num', 'feature_num', 'location']]

# merge_set = dt_train.merge(dt_pn, how='inner', on='pn_num').reset_index()
# merge_set.to_csv(root + 'test.csv', index=False)
rule = re.compile('[\d\s;,]')
# df_preprocessing = df_train.merge(df_pn, how='inner', on='pn_num').reset_index().to_dict()
df_preprocessing = df_train.merge(df_pn, how='inner', on='pn_num').reset_index()
df_preprocessing = df_preprocessing[df_preprocessing.location != '[]']
df_preprocessing.to_csv(root + 'check.csv', index=False)
loc_dict = df_train['location'].to_dict()
# print(merge_set_dict)
# loc_list = ''.join(re.findall(rule, merge_set['location'].to_dict()[3]))
# for keys in loc_dict:
#     loc = []
#     if bool(rule.search(loc_dict[keys])):
#         valid_loc = rule.findall(loc_dict[keys])
#         valid_loc = ''.join(valid_loc).split(',')
#         print(valid_loc)
#         for loc_idx in valid_loc:
#             if ';' not in loc_idx:
#                 loc_idx = loc_idx.strip().split(' ')
#                 print(loc_idx)
#                 loc.append([int(loc_idx[0]), int(loc_idx[1])])
#             else:
#                 loc_idx = loc_idx.split(';')
#                 for sub_idx in loc_idx:
#                     sub_idx = sub_idx.strip().split(' ')
#                     loc.append([int(sub_idx[0]), int(sub_idx[1])])
#         print(loc)
#         df_preprocessing['location'][keys] = loc
#     else:
#         del df_preprocessing['pn_num'][keys]
#         del df_preprocessing['location'][keys]
#         del df_preprocessing['index'][keys]
#         del df_preprocessing['feature_num'][keys]

print(df_preprocessing)

df_loc = pd.DataFrame.from_dict(df_preprocessing)
# dt_loc['pn_num'] = dt_loc.index
# print(df_loc)
# dt_loc.to_csv(root + 'dt_loc.csv', index=False)
# train = df_loc.merge(df_pn, how='inner', on='pn_num').reset_index()
# train.to_csv(root + 'train.csv', index=False)
# print(train)
# print(merge_set['location'].to_dict()[3])
# print(''.join(re.findall(rule, merge_set['location'].to_dict()[3])))

