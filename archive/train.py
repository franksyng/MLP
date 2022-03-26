import os
import utils
from model import *
from transformers import BertTokenizer


def train(epoch_num):
    model.train()
    for epoch in range(epoch_num):
        for i, data in enumerate(training_loader, 0):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)
            # criterion = nn.CrossEntropyLoss()
            loss = model(ids, mask, labels=targets)[0]
            # output = model(ids, mask)
            # loss = criterion(output, targets)
            if i % 500 == 0 and i != 0:
                print(f'Epoch {epoch} finished with loss {loss.item()}')
            elif i == 0:
                print(f'Initial loss: {loss.item()}')
            else:
                print(f'epoch: {epoch}, iter: {i}, loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), os.path.join(root, "checkpoint/model" + str(epoch) + "_.pt"))
            # xm.optimizer_step(optimizer)
            # xm.mark_step()


if __name__ == "__main__":
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
    EPOCHS = 10
    LEARNING_RATE = 0.001
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

    model = BERT()
    model.to(dev)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    train(EPOCHS)
