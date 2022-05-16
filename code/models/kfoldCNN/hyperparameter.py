from CNN import CNN
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class hyperparameter_CNN:

    def __init__(self):
        self.model = CNN()


    def check_accuracy(self, validation_loader):

        pred_list = []
        y_list = []
        for i,(x, y) in enumerate(validation_loader):

            scores = self.model(x)
            _, predictions = scores.max(1)

            pred_list+=predictions.tolist()
            y_list+=y.tolist()
        f1 = f1_score(y_list, pred_list, average='macro')
        acc = accuracy_score(y_list, pred_list)

        confs_matrix = confusion_matrix(y_list, pred_list)
        cm = confs_matrix.astype('float') / confs_matrix.sum(axis=1)[:, np.newaxis].tolist()
        return acc, f1, cm.tolist()

    def softmax(self, validation_loader):

        all_scores = []
        for i,(x, y) in enumerate(validation_loader):

            scores = self.model(x)
            all_scores += scores.tolist()

        return all_scores

    def save_model(self, path):

        torch.save(self.model, path)



    def train(self, lr, momentum, train_loader, epochs):

        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        # optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.8)

        running_loss_list = []
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images)
                labels = Variable(labels)

                optimizer.zero_grad()
                # print(images.shape)
                outputs = self.model(images)
                # print(labels)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:    # print every 2000 mini-batches
                    running_loss_list.append(running_loss)
                    # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0
        return running_loss_list

    def run(self, lr, momentum, val_loader, train_loader, epochs):

        self.train(lr, momentum, train_loader, epochs)
        acc, f1, _ = self.check_accuracy(val_loader)

        return acc,f1