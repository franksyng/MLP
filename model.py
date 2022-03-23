from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import AutoTokenizer, AutoModel, BertForTokenClassification


class SentenceGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            # padding='max_length',
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.labels[index]
        label.extend([4] * 200)
        label = label[:200]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len


class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        # self.layer = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=278)
        self.layer = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=278)
        # self.dropout = torch.nn.Dropout(0.3)
        # self.fc = torch.nn.Linear(768, 200)

    def forward(self, ids, mask, labels=None):
        output_1 = self.layer(ids, mask, labels=labels)
        # output_2 = self.l2(output_1[1])
        # output = self.l3(output_2)
        # output_1 = self.layer(ids, mask)
        # output_2 = self.dropout(output_1[0])
        # output = self.fc(output_1[1])
        return output_1
