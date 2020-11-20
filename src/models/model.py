import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layer, label_size):  # , batch_size)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layer)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim), 
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs
