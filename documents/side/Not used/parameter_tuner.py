from sklearn.model_selection import GridSearchCV
import pandas as pd
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, in_channels=1, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2,2), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride= (2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=(2,2), stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(576, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


def main():
    model = CNN()





if __name__=="__main__":
    main()

