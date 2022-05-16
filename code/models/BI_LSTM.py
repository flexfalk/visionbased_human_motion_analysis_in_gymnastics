import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random

torch.manual_seed(1)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size, num_layers=2):
        super(LSTMTagger, self).__init__()
        #super(LSTMTagger, self).__init__(num_layers)
        #super().__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        # self.lstm_bi = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2, bidirectional=True)
        # self.lstm2 = nn.LSTM(hidden_dim, hidden_dim2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)



    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        # lstm_out, _ = self.lstm_bi(sentence.view(len(sentence), 1, -1))

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        #print(sentence)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def prepare_sequence_X(self, seq):
        return torch.tensor(seq, dtype=torch.float, requires_grad=True)

    def prepare_sequence_Y(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)