import torch.nn as nn
import torch
from torch.utils.data import Dataset
import time
import pkuseg


class LstmNet(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(LstmNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1)
        self.predictor = nn.Linear(hidden_size, 17)

    def forward(self, seq):
        output, (hidden,_) = self.encoder(self.embedding(seq).permute(1,0,2))
        preds = self.predictor(hidden.squeeze(0))
        return preds.squeeze(0)


class LstRun:
    def __init__(self, epochs, model, optimizer, loss_fn, param_path, labels):
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.labels = labels
        self.param_path = param_path
        state_dict = torch.load(param_path)
        self.model.load_state_dict(state_dict)

    def train(self, train_iterator, valid_iterator):
        time_all_start = time.time()
        for epoch in range(self.epochs):
            time_start = time.time()
            training_loss = 0.0
            self.model.train()
            for batch in train_iterator:
                self.optimizer.zero_grad()
                text = batch["text"]
                label = batch["label"]
                output = self.model(text)
                loss = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.data.item()
            training_loss /= len(train_iterator)
            time_train = time.time()
            print(f"epoch:{epoch},training loss:{training_loss}, "
                  f"time-cost: {time_train-time_start}")
            self.model.eval()
            valid_loss = 0
            correct_num = 0
            example_num = 0
            for batch in valid_iterator:
                text = batch["text"]
                label = batch["label"]
                output = self.model(text)
                loss2 = self.loss_fn(output, label)
                valid_loss += loss2.data.item()
                correct = torch.eq(torch.max(torch.nn.functional.softmax(output), dim=1)[1], label)
                correct_num += torch.sum(correct)
                example_num += correct.shape[0]
            valid_loss /= len(valid_iterator)
            accuracy = correct_num / example_num
            time_valid = time.time()

            print(f"epoch:{epoch},validation loss:{valid_loss},"
                  f"accuracy:{accuracy}, time-cost: {time_valid-time_train}")

            torch.save(self.model.state_dict(), self.param_path)
        time_all_end = time.time()
        print(f"total-time-cost: {time_all_end-time_all_start}")

    def predict(self, sentence, top=1):

        sentence = sentence.unsqueeze(0)

        prediction = self.model(sentence)
        prediction_list = []
        for i in range(top):
            max_index = prediction.argmax().item()
            prediction_list.append(self.labels[max_index])
            prediction[max_index] = torch.tensor([0])
        return prediction_list


class DataFrameDataset(Dataset):
    def __init__(self, content, label):
        self.content = content
        self.label = label

    def __getitem__(self, index):
        return self.content[index], self.label[index]

    def __len__(self):
        return len(self.label)


def collate_batch(data):
    unit_x = []
    unit_y = []
    for unit in data:
        unit_x.append(unit[0])
        unit_y.append(unit[1]-100)
    return {"text": torch.tensor(unit_x),
            "label": torch.tensor(unit_y)}


def convert_sentence(sentence, vocab, max_size):
    seg = pkuseg.pkuseg()
    sen_list = seg.cut(sentence)
    sen_len = len(sen_list)
    if sen_len > max_size:
        sen_list = sen_list[:max_size]
    elif sen_len < max_size:
        sen_list.extend(["<pad>"]*(max_size - sen_len))
    for index, sen_part in enumerate(sen_list):
        if sen_part in vocab.keys():
            sen_list[index] = vocab[sen_part]
        else:
            sen_list[index] = 0
    return torch.tensor(sen_list)

