import torch.nn as nn
from ScdmsML.src.utils import ListModule


class NeuralNetwork(nn.Module):
    def __init__(self, sizes: list):
        super(NeuralNetwork, self).__init__()
        self.sizes = sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())
        del layers[-1]
        layers.append(nn.Sigmoid())
        self.layers = ListModule(*layers)

        self.float()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




