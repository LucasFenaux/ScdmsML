import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x):
        return self.layer(x)
