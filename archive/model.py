import tokenizers.decoders
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
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
        label = self.labels[index]
        new_label = []
        previous_word_idx = None
        tokenized_inputs = self.tokenizer(sentence.split(' '), truncation=True, max_length=self.max_len,
                                          padding='max_length', is_split_into_words=True)
        decode_tkn = self.tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'])
        word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                new_label.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                new_label.append(label[word_idx])
            else:
                new_label.append(label[previous_word_idx])
            previous_word_idx = word_idx
        ids = tokenized_inputs['input_ids']
        mask = tokenized_inputs['attention_mask']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(new_label, dtype=torch.long),
            'tkn_result': decode_tkn
        }

    def __len__(self):
        return self.len


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.layer = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').eval()
        # self.layer = AutoModel.from_pretrained('transformersbook/bert-base-uncased-finetuned-clinc', num_labels=278)
        # self.layer = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.fc = nn.Linear(768, 287)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, ids, mask, labels=None):
        with torch.no_grad():
            output = self.layer(ids, mask)[0]
        output = self.dropout(output)
        output = self.fc(output)
        output = self.relu(output)
        return output


class BERT_LSTM_CNN(nn.Module):
    def __init__(self, hdim=768):
        super(BERT_LSTM_CNN, self).__init__()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_num = 287
        self.bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').eval()
        self.lstm = nn.LSTM(768, hdim, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(hdim, 287, 3, padding=1),
            nn.ReLU(),
        )
        # self.maxpool = nn.MaxPool1d(kernel_size=3)

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.dev),
              torch.randn(2, batch_size, self.hidden_dim // 2).to(self.dev))

    def forward(self, ids, mask, labels=None):
        with torch.no_grad():
            embeds = self.bert(ids, mask)[0]  # [32, maxlen, 768]
        lstm_out, _ = self.lstm(embeds)  # [32, maxlen, 768]
        conv_out = self.conv(torch.permute(lstm_out, (0, 2, 1)))
        return torch.permute(conv_out, (0, 2, 1))
        # print(conv_out.size())
        # maxpool = self.maxpool(conv_out)
        # print(maxpool.size())


class LSTM_CLS(nn.Module):
    def __init__(self, class_num, hidden_dim=768):
        super(LSTM_CLS, self).__init__()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.target_num = class_num
        self.scibert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(self.dev)
        self.lstm = nn.LSTM(768, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.fc = nn.Linear(hidden_dim, self.target_num)
        self.hidden = self.init_hidden(batch_size=32)

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.dev),
              torch.randn(2, batch_size, self.hidden_dim // 2).to(self.dev))

    def _bert_enc(self, ids, mask):
        """
        x: [batchsize, sent_len]
        enc: [batch_size, sent_len, 768]
        """
        with torch.no_grad():
            enc = self.scibert(ids, mask)[0]
        return enc

    def forward(self, ids, mask, batch_size):
        self.hidden = self.init_hidden(batch_size=batch_size)
        embeds = self._bert_enc(ids, mask)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        outputs = self.fc(lstm_out)
        return outputs


