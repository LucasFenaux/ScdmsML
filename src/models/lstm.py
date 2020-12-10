from typing import Any

import torch.nn as nn
import torch

# Modified source: https://www.kaggle.com/purplejester/a-simple-lstm-based-time-series-classifier


class OldLSTMClassifier(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.sigmoid = nn.Sigmoid()
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        #print(out)
        #out = self.sigmoid(out)
        return out[:, 0]

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # we return cuda because we assume you're are running this on gpu
        # if not, please don't bother unless you want to waste your time
        return [t.cuda() for t in (h0, c0)]
