import argparse
import os
import utils
import torch
from model import *
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer
from seqeval.metrics import f1_score
import numpy as np


def flat_accuracy(preds, labels):
    flat_preds = np.argmax(preds, axis=2).flatten()
    flat_labels = labels.flatten()
    return np.sum(flat_preds == flat_labels)/len(flat_labels)


def valid(model, testing_loader):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    n_correct = 0; n_wrong = 0; total = 0
    predictions , true_labels = [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(dev, dtype = torch.long)
            mask = data['mask'].to(dev, dtype = torch.long)
            targets = data['tags'].to(dev, dtype = torch.long)

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


def train(epoch_num):
    model.train()
    best_loss = 10
    best_epoch = 0
    for epoch in range(epoch_num):
      print(f'Epoch: {epoch}')
      for i, data in enumerate(training_loader, 0):
          ids = data['ids'].to(dev, dtype=torch.long)
          mask = data['mask'].to(dev, dtype=torch.long)
          targets = data['tags'].to(dev, dtype=torch.long)
          # criterion = nn.CrossEntropyLoss()
          loss = model(ids, mask, labels=targets)[0]
          curr_loss = loss.item()
          # output = model(ids, mask)
          # loss = criterion(output, targets)
          if i % 500 == 0 and i == 0:
              print(f'Initial loss of epoch {epoch} is {loss.item()}')
          else:
              print(f'iter: {i}, loss:  {loss.item()}')

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      if curr_loss < best_loss:
        best_loss = curr_loss
        best_epoch = epoch
        torch.save(model, os.path.join(root, "checkpoint/best_model.pt"))
      print(f'Epoch {epoch} finished - loss: {loss.item()}, best epoch: {best_epoch}, best loss: {best_loss}')
          # xm.optimizer_step(optimizer)
          # xm.mark_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--ckpt', default=None)
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
    tags_vals = list(set(dataset["tag"].values))
    tag2idx = {t: i for i, t in enumerate(tags_vals)}
    sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]
    # sentences = [[s[0] for s in sent] for sent in getter.sentences]
    labels = [[s[1] for s in sent] for sent in getter.sentences]
    labels = [[tag2idx.get(l) for l in lab] for lab in labels]

    # ========= Training variables ========== #
    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # ========= Creating the dataset and dataloader for the neural network ========== #
    train_percent = 0.8
    train_size = int(train_percent * len(sentences))
    # train_dataset=df.sample(frac=train_size,random_state=200).reset_index(drop=True)
    # test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_sentences = sentences[0:train_size]
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
                    'shuffle': False,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    if args.mode == 'train':
        model = BERT()
        model.to(dev)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        train(EPOCHS)
    elif args.mode == 'test':
        model_path = args.ckpt
        model = torch.load(model_path)
        valid(model, testing_loader)
