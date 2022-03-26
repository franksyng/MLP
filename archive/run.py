import argparse
import os
import utils
from model import *
import torch.nn as nn
import torch.nn.functional
from transformers import AutoTokenizer
# from seqeval.metrics import f1_score
import numpy as np
from sys import stdout
from sklearn.metrics import classification_report


def cal_accuracy(preds, label_ids, mask):
    valid_len = np.sum(mask)
    flat_preds = preds.to('cpu').numpy().flatten()[:valid_len]
    flat_labels = label_ids.flatten()[:valid_len]
    acc = classification_report(flat_labels, flat_preds, output_dict=True)['accuracy']
    new_labels = [i for i in flat_labels if i != 286]
    new_labels = list(dict.fromkeys(new_labels))
    target_names = [str(i) for i in new_labels]
    ner_f1 = classification_report(flat_labels, flat_preds, labels=new_labels, target_names=target_names, output_dict=True)['f1-score']
    return acc, ner_f1


def validation(model, testing_loader, model_name='LSTM_CLS', batch_size=None):
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    eval_ner_acc = 0
    n_correct = 0
    n_wrong = 0
    total = 0
    predictions, true_labels = [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)
            if model_name == 'LSTM_CLS':
                output = model(ids, mask, batch_size)
            else:
                output = model(ids, mask)
            loss = criterion(torch.transpose(output, 1, 2), targets)
            preds = nn.functional.softmax(output, dim=2)
            preds = torch.argmax(preds, dim=2)
            label_ids = targets.to('cpu').numpy()
            true_labels.append(label_ids)
            accuracy, ner_accuracy = cal_accuracy(preds, label_ids, mask.to('cpu').numpy())
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            eval_ner_acc += ner_accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        stdout.write("Validation loss: {}\n".format(eval_loss))
        stdout.write("Validation Accuracy: {}\n".format(eval_accuracy/nb_eval_steps))
        stdout.write("Validation NER f1-score: {}\n".format(eval_ner_acc / nb_eval_steps))
        stdout.flush()
        # pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        # valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        # print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


def train(epoch_num, batch_size, model_name='LSTM_CLS'):
    best_avg_loss = 10
    best_epoch = 0
    for epoch in range(epoch_num):
        model.train()
        cumulative_loss = []
        curr_avg_loss = 0
        for i, data in enumerate(training_loader, 0):
            iter_total = len(training_loader)
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)  # [32, 200]
            model.zero_grad()
            if model_name == 'LSTM_CLS':
                output = model(ids, mask, batch_size)
            else:
                output = model(ids, mask)
            loss = criterion(torch.transpose(output, 1, 2), targets)
            curr_loss = loss.item()
            cumulative_loss.append(curr_loss)
            curr_avg_loss = sum(cumulative_loss) / len(cumulative_loss)
            if i == 0:
                stdout.write(f'======== {model_name}: Starting epoch {epoch} ========\n')
                stdout.write(f'[{i + 1}/{iter_total}] - initial loss:  {loss.item()}\n')
            elif (i + 1) % batch_size == 0:
                # stdout.write(f'[{i + 1}/{iter_total}] - loss:  {loss.item()} ({curr_avg_loss})\n')
                stdout.write(f'[{i + 1}/{iter_total}] - loss: {curr_avg_loss}\n')
            stdout.flush()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if curr_avg_loss < best_avg_loss:
            best_avg_loss = curr_avg_loss
            best_epoch = epoch
            torch.save(model, os.path.join(root, "checkpoint/best_model.pt"))
        stdout.write(f'Epoch {epoch} finished - avg. loss: {curr_avg_loss}, best epoch: {best_epoch}, best loss: {best_avg_loss}\n')
        stdout.flush()
        validation(model, testing_loader, model_name, int(batch_size/2))
        # xm.optimizer_step(optimizer)
        # xm.mark_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--model', default='LSTM_CLS')
    parser.add_argument('--epoch', default=30)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--max_len', default=250)
    parser.add_argument('--lr', default=0.001)

    args = parser.parse_args()

    root = ''
    data_root = 'data'
    data_path = os.path.join(data_root, 'train.csv')
    pn_path = os.path.join(data_root, 'patient_notes.csv')
    feature_path = os.path.join(data_root, 'features.csv')
    preprocessor = utils.Preprocessor(data_path, pn_path, feature_path)
    dataset = preprocessor.to_dataframe()
    getter = SentenceGetter(dataset)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dev = xm.xla_device()

    # ========= Tag to idx ========== #
    tag2idx = preprocessor.make_vocab()
    sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]
    # sentences = [s[0] for sent in getter.sentences for s in sent]
    # print(sentences)
    labels = [[s[1] for s in sent] for sent in getter.sentences]
    labels = [[tag2idx.get(l) for l in lab] for lab in labels]

    # ========= Training variables ========== #
    MAX_LEN = 250
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 0.001
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # tokenizer = AutoTokenizer.from_pretrained('transformersbook/bert-base-uncased-finetuned-clinc')
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # ========= Creating the dataset and dataloader for the neural network ========== #
    train_percent = 0.8
    train_size = int(train_percent * len(sentences))
    # train_dataset=df.sample(frac=train_size,random_state=200).reset_index(drop=True)
    # test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_sentences = sentences[0:train_size]
    # print(train_sentences)
    train_labels = labels[0:train_size]

    test_sentences = sentences[train_size:]
    test_labels = labels[train_size:]

    print("FULL Dataset: {}".format(len(sentences)))
    print("TRAIN Dataset: {}".format(len(train_sentences)))
    print("TEST Dataset: {}".format(len(test_sentences)))

    training_set = CustomDataset(tokenizer, train_sentences, train_labels, MAX_LEN)
    testing_set = CustomDataset(tokenizer, test_sentences, test_labels, MAX_LEN)

    # ========= Parameters ========== #
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(training_set, **test_params)

    # ========= Char embedding ========== #
    embeds = utils.char_embedding(training_loader)

    # optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if args.mode == 'train':
        if args.model == 'BERT':
            model = BERT()
        elif args.model == 'BERT_LSTM_CNN':
            model = BERT_LSTM_CNN()
        elif args.model == 'LSTM_CLS':
            model = LSTM_CLS(287)
        else:
            model = None
        model.to(dev)
        # optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.9)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
        train(EPOCHS, TRAIN_BATCH_SIZE, args.model)
    elif args.mode == 'test':
        model_path = args.ckpt
        # model = torch.load(model_path, map_location=torch.device('cpu'))
        model = torch.load(model_path)
        validation(model, testing_loader)
