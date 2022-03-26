import argparse
import os
import utils
from model import *
import random
import torch.nn as nn
import torch.nn.functional
from transformers import AutoTokenizer
import numpy as np
from sys import stdout
import warnings
warnings.filterwarnings('ignore')


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
            accuracy, ner_accuracy = utils.cal_accuracy(preds, label_ids, mask.to('cpu').numpy())
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
        for i, data in enumerate(train_loader, 0):
            iter_total = len(train_loader)
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
        validation(model, val_loader, model_name, batch_size)
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
    parser.add_argument('--lr', default=0.0001)
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

    # ========= Make dataset trainable ========== #
    getter_sentences = getter.sentences

    # random.shuffle(getter_sentences)
    tag2idx = preprocessor.make_vocab()
    sentences, labels = utils.sentence_and_label(getter_sentences, tag2idx)
    label_stat = utils.count_label(labels)
    max_num = label_stat[143]

    # ========= Split dataset ========== #
    train_percent = 0.8
    val_percent = 0.2
    test_percent = 1 - train_percent - val_percent

    test_data = None
    test_labels = None
    if train_percent + val_percent == 1:
        train_split, val_split = utils.split_dataset(sentences, labels, train_percent, val_percent)
    else:
        test_size = int(test_percent * len(sentences))
        train_split, val_split, test_split = utils.split_dataset(sentences, labels, train_percent, val_percent, test_percent)
        test_data = test_split[0]
        test_labels = test_split[1]

    train_data = train_split[0]
    train_labels = train_split[1]
    val_data = val_split[0]
    val_labels = val_split[1]

    stdout.write("Full Dataset: {}\n".format(len(sentences)))
    stdout.write("Train Dataset: {}\n".format(len(train_data)))
    stdout.write("Validation Dataset: {}\n".format(len(val_data)))
    if test_data is not None:
        stdout.write("Test Dataset: {}\n".format(len(test_data)))
    stdout.flush()

    # ========= Form tokenized dataset ========== #
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    train_set = CustomDataset(tokenizer, train_data, train_labels, args.max_len)
    val_set = CustomDataset(tokenizer, val_data, val_labels, args.max_len)

    # ========= Data loader ========== #
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    val_params = {'batch_size': args.batch_size,
                  'shuffle': True,
                  'num_workers': 0
                  }

    test_params = {'batch_size': args.batch_size,
                   'shuffle': True,
                   'num_workers': 0
                   }

    train_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # ========= Char embedding ========== #
    embeds = utils.char_embedding(train_loader)

    # ========= NN dependencies ========== #
    class_weights = torch.FloatTensor([label_stat[i] if i in label_stat.keys() else 0 for i in range(287)]).to(dev)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.batch_size)
        train(int(args.epoch), args.batch_size, args.model)







