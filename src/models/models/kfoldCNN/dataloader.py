from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from json_to_csv_clone import json_to_csv

class SkeletonData(data.Dataset):

    def __init__(self, path, json=False):
        super(SkeletonData, self).__init__()

        if json:
            j_t_c = json_to_csv(path)
            data = j_t_c.make_df()

        if not json:
            data = pd.read_csv(path)



        data["finals"][data["finals"]=="idle"] = 0
        data["finals"][data["finals"]=="take-off"] = 1
        data["finals"][data["finals"]=="skill"] = 2
        data["finals"][data["finals"]=="landing"] = 3
        # data.finals = pd.factorize(data.finals)[0]
        data_labels = data['finals'].values
        data = data.drop(["clipname", "videoname","finals"],axis=1).values.reshape(len(data), 1, 2, 33)


        self.datalist = data
        self.labels = data_labels

    def __getitem__(self, index):
        return torch.Tensor(self.datalist[index].astype(float)), self.labels[index]



    def __len__(self):
        return self.datalist.shape[0]



