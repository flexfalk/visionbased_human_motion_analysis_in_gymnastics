from code.models.BI_LSTM import LSTMTagger as LSTM
from sklearn.metrics import f1_score, accuracy_score
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.metrics import confusion_matrix
EMBEDDING_DIM = 66
HIDDEN_DIM = 4
HIDDEN_DIM2 = 4

class hyperparameter_LSTM:

    def __init__(self):
        self.tag_to_ix = {"idle": 0, "take-off": 1, "skill": 2, "landing": 3}
        self.model = LSTM(embedding_dim=66, hidden_dim=4, tagset_size=4)

    def check_accuracy(self, validation_data:list):

        self.model.eval()

        pred_list = []
        y_list = []
        for i, (x, y) in enumerate(validation_data):

            inputs = self.model.prepare_sequence_X(x)

            labels = self.model.prepare_sequence_Y(y, self.tag_to_ix)

            tag_scores = self.model(inputs)

            for ix, frame in enumerate(tag_scores):
                frame_label = torch.argmax(frame)
                pred_list.append(int(frame_label))
                y_list.append(labels[ix])

        f1 = f1_score(y_list, pred_list, average='macro')
        acc = accuracy_score(y_list, pred_list)

        confs_matrix = confusion_matrix(y_list, pred_list)
        cm = confs_matrix.astype('float') / confs_matrix.sum(axis=1)[:, np.newaxis]

        return acc, f1, cm.tolist()

    def softmax(self, validation_data):

        all_scores = []

        for i, (x,y) in enumerate(validation_data):

            inputs = self.model.prepare_sequence_X(x)

            scores = self.model(inputs)
            all_scores += scores.tolist()

        return all_scores

    def save_model(self, path):

        torch.save(self.model, path)

    def train(self, train_data:list, epochs, beta1=None, beta2=None, lr = None, momentum= None):

        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        # optimizer = optim.Adam(self.model.parameters(), betas=(beta1, beta2))
        running_loss_list = []

        for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
            running_loss = 0.0
            self.model.train()
            for i, (sentence, tags) in enumerate(train_data):
                self.model.zero_grad()
                sentence_in = self.model.prepare_sequence_X(sentence)

                targets = self.model.prepare_sequence_Y(tags, self.tag_to_ix)
                tag_scores = self.model(sentence_in)

                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:  # print every 2000 mini-batches
                    running_loss_list.append(running_loss)
                    # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0

        return running_loss_list

    def run(self, val_data, train_data, epochs, beta1=None, beta2=None, lr = None, momentum= None):

        self.train(train_data, epochs, lr=lr, momentum=momentum)
        acc, f1, _ = self.check_accuracy(val_data)

        return acc, f1