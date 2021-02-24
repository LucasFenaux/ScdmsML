import torch
import torch.nn as nn
import numpy as np


class LSTMClassifier(nn.Module):
    """
    Version V1.3
    A regular one layer LSTM classifier using LSTMCell -> FFNetwork structure
    """
    def __init__(self, input_dim, hidden_dim, label_size, device=torch.device("cuda"), dropout_rate=0.1):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.hidden2ff = nn.Linear(hidden_dim,  int(np.sqrt(hidden_dim)))
        self.ff2label = nn.Linear(int(np.sqrt(hidden_dim)), label_size)
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
        for name, param in self.hidden2ff.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.ff2label.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):

        hs = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cs = torch.zeros(x.size(0), self.hidden_dim).to(self.device)

        for i in range(x.size()[1]):
            hs, cs = self.lstm(x[:, i], (hs, cs))
            # hs = self.dropout(hs)
            # cs = self.dropout(cs)
        hs = self.dropout(hs)
        hs = self.hidden2ff(hs)
        # return self.sigmoid(self.hidden2label(hs.reshape(x.shape[0], -1)))
        return self.sigmoid(self.ff2label(hs))


class BiLSTMClassifier(nn.Module):
    """
    Version V0.2
    A bidirectional LSTM using a bidirectional layer followed by a normal LSTM layer and
    ending with a FFNetwork
    """
    def __init__(self, input_dim, hidden_dim, label_size, device=torch.device("cuda"), dropout_rate=0.1):
        super().__init__()
        self.forwards = nn.LSTMCell(input_dim, hidden_dim)
        self.backwards = nn.LSTMCell(input_dim, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim*2, hidden_dim*2)
        self.hidden2ff = nn.Linear(hidden_dim*2,  int(np.sqrt(2*hidden_dim)))
        self.ff2label = nn.Linear(int(np.sqrt(2*hidden_dim)), label_size)
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
        for name, param in self.hidden2ff.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.ff2label.named_parameters():
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
            # hf = self.dropout(hf)
            # cf = self.dropout(cf)
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
        hs = self.dropout(hs)
        hs = self.hidden2ff(hs)
        return self.sigmoid(self.ff2label(hs))


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


class CNN_LSTM_Classifier(nn.Module):
    """
    Version V0.2
    CNN+LSTM classifier using ConvNet -> LSTMCell -> FFNetwork structure
    """
    def __init__(self, input_dim, hidden_dim, label_size, device=torch.device("cuda"), dropout_rate=0.1):
        super().__init__()
        self.convnet1 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=4,
                                               padding=2), nn.MaxPool1d(kernel_size=2),
                                     nn.BatchNorm1d(input_dim), nn.ReLU()) # here to reduce noice
        self.convnet2 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=2*input_dim, kernel_size=4,
                                               padding=2), nn.MaxPool1d(kernel_size=2, stride=2),
                                     nn.BatchNorm1d(2*input_dim), nn.ReLU())  # start extracting features and reduce size
                                                                              # ro reduce complexity for lstm
        self.convnet3 = nn.Sequential(nn.Conv1d(in_channels=2*input_dim, out_channels=4*input_dim, kernel_size=4,
                                               padding=2), nn.MaxPool1d(kernel_size=2),
                                     nn.BatchNorm1d(4*input_dim), nn.ReLU())
        # removed to reduce model complexity to try and limit overfitting
        # self.convnet4 = nn.Sequential(nn.Conv1d(in_channels=4*input_dim, out_channels=8*input_dim, kernel_size=4,
        #                                        padding=2), nn.BatchNorm1d(8*input_dim), nn.ReLU())
        # self.convnet5 = nn.Conv1d(in_channels=8*input_dim, out_channels=16*input_dim, kernel_size=4, padding=2)
        # self.lstm = nn.LSTMCell(16*input_dim, hidden_dim)
        self.convnet5 = nn.Conv1d(in_channels=4*input_dim, out_channels=8*input_dim, kernel_size=4, padding=2)
        self.lstm = nn.LSTMCell(8*input_dim, hidden_dim)
        self.hiddentoff = nn.Linear(hidden_dim, int(np.sqrt(hidden_dim)))
        self.relu = nn.ReLU()
        self.fftolabel = nn.Linear(int(np.sqrt(hidden_dim)), label_size)
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
        for name, param in self.hiddentoff.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.fftolabel.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        x = torch.reshape(x, (x.size()[0], x.size()[2], x.size()[1]))
        x = self.convnet1(x)
        x = self.convnet2(x)
        x = self.convnet3(x)
        # x = self.convnet4(x)
        x = self.convnet5(x)
        x = torch.reshape(x, (x.size()[0], x.size()[2], x.size()[1]))
        x = self.dropout(x)
        hs = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cs = torch.zeros(x.size(0), self.hidden_dim).to(self.device)

        for i in range(x.size()[1]):
            hs, cs = self.lstm(x[:, i], (hs, cs))
            # hs = self.dropout(hs)
            # cs = self.dropout(cs)

        hs = self.dropout(hs)
        hs = self.hiddentoff(hs)
        hs = self.relu(hs)
        # return self.sigmoid(self.hidden2label(hs.reshape(x.shape[0], -1)))
        return self.sigmoid(self.fftolabel(hs))
