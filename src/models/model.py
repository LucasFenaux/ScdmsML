import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Version V1.2
    A regular one layer LSTM classifier using LSTMCell -> FFNetwork structure
    """
    def __init__(self, input_dim, hidden_dim, label_size, device=torch.device("cuda"), dropout_rate=0.1):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.hidden2label.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):

        hs = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cs = torch.zeros(x.size(0), self.hidden_dim).to(self.device)

        for i in range(x.size()[1]):
            hs, cs = self.lstm(x[:, i], (hs, cs))
            hs = self.dropout(hs)
            cs = self.dropout(cs)

        # return self.sigmoid(self.hidden2label(hs.reshape(x.shape[0], -1)))
        return self.sigmoid(self.hidden2label(hs))


class BiLSTMClassifier(nn.Module):
    """
    Version V0.1
    A bidirectional LSTM using a bidirectional layer followed by a normal LSTM layer and
    ending with a FFNetwork
    """
    def __init__(self, input_dim, hidden_dim, label_size, device=torch.device("cuda"), dropout_rate=0.1):
        super().__init__()
        self.forwards = nn.LSTMCell(input_dim, hidden_dim)
        self.backwards = nn.LSTMCell(input_dim, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim*2, hidden_dim*2)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.forwards.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.backwards.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.hidden2label.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        # Initializing hidden states
        hf = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cf = torch.zeros(x.size(0), self.hidden_dim).to(self.device)

        hb = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cb = torch.zeros(x.size(0), self.hidden_dim).to(self.device)

        hs = torch.zeros(x.size(0), self.hidden_dim*2).to(self.device)
        cs = torch.zeros(x.size(0), self.hidden_dim*2).to(self.device)
        # Performing Forward Pass
        f = []
        for i in range(x.size()[1]):
            hf, cf = self.forwards(x[:, i], (hf, cf))
            hf = self.dropout(hf)
            cf = self.dropout(cf)
            f.append(hf)
        # Performing Backward Pass
        b = []
        for i in reversed(range(x.size()[1])):
            hb, cb = self.backwards(x[:, i], (hb, cb))
            hb = self.dropout(hb)
            cb = self.dropout(cb)
            b.append(hb)
        # Performing LSTM Pass
        for fwd, bwd in zip(f,b):
            input_tensor = torch.cat((fwd, bwd), 1)
            hs, cs = self.lstm(input_tensor, (hs, cs))
            # maybe add dropout here as well if necessary

        return self.sigmoid(self.hidden2label(hs))


class FFClassifier(nn.Module):
    """ V0.1 of a classic FeedForward Classifying network """
    def __init__(self, dropout_rate):
        super(FFClassifier, self).__init__()
        self.nn = nn.Sequential(nn.Linear(4096, 2048), nn.LeakyReLU(), nn.Dropout(dropout_rate), nn.Linear(2048, 1024),
                                nn.LeakyReLU(), nn.Dropout(dropout_rate), nn.Linear(1024, 512),
                                nn.LeakyReLU(), nn.Dropout(dropout_rate), nn.Linear(512, 256),
                                nn.LeakyReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 100),
                                nn.LeakyReLU(), nn.Dropout(dropout_rate), nn.Linear(100, 30),
                                nn.LeakyReLU(), nn.Dropout(dropout_rate), nn.Linear(30, 2), nn.Sigmoid())

        self.dropout_rate = dropout_rate
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.nn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0002)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        return self.nn(x)
