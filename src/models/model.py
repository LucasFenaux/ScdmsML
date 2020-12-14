import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_size, num_layers, batch_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        _, (h_n, _) = self.lstm(x, (h0, c0))
        return self.sigmoid(self.hidden2label(h_n.reshape(x.shape[0], -1))[:, 0])


# import torch.nn as nn
# import torch
# import torch.nn.functional as F
#
#
# class LSTMClassifier(nn.Module):
#
#     def __init__(self, embedding_dim, hidden_dim, num_layer, label_size, batch_size):
#         super(LSTMClassifier, self).__init__()
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layer)
#         self.hidden2label = nn.Linear(hidden_dim, label_size)
#         self.batch_size = batch_size
#         self.hidden_dim = hidden_dim
#         self.hidden = self.init_hidden()
#
#     def init_hidden(self):
#         return (torch.zeros(1, self.batch_size, self.hidden_dim),
#                 torch.zeros(1, self.batch_size, self.hidden_dim))
#
#     def forward(self, x):
#         lstm_out, self.hidden = self.lstm(x, self.hidden)
#         y = self.hidden2label(lstm_out[-1])
#         log_probs = F.log_softmax(y)
#         return log_probs
