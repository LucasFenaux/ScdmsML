from __future__ import absolute_import
import sys
import os
sys.path.insert(0, '/home/fenauxlu/projects/rrg-mdiamond/fenauxlu/ScdmsML/src/')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.model import LSTMClassifier
import torch
import torch.optim as optim
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


if __name__ == '__main__':
    device = torch.device('cpu')
    train = pd.read_csv('../input/train_1.csv').fillna(0)
    page = train['Page']
    train.head()
    # Dropping Page Column
    train = train.drop('Page', axis=1)
    # Using Data From Random Row for Training and Testing

    row = train.iloc[90000, :].values
    X = row[0:164]
    y = row[165]
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler()
    X_train = np.reshape(X_train, (-1, 1))
    y_train = np.reshape(y_train, (-1, 1))
    X_train = sc.fit_transform(X_train)
    X_train = np.reshape(X_train, (165, 1, 1))
    y_train = sc.fit_transform(y_train)

    input_size = 1
    hidden_size = 8
    label_size = 1
    batch_size = 10
    epochs = 100

    criterion = torch.nn.MSELoss()
    regressor = LSTMClassifier(input_size, hidden_size, label_size, 1, 165)
    regressor = regressor.to(device)
    learning_rate = 0.01
    optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)

    trainer = create_supervised_trainer(regressor, optimizer, criterion, device=device)

    # Setting up the dataset
    train_data = torch.Tensor(X_train)
    train_targets = torch.Tensor(y_train)

    train_dataset = TensorDataset(train_data, train_targets)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=1,
                              pin_memory=False)

    trainer.run(train_loader, max_epochs=epochs)

    # Getting the predicted Web View
    inputs = X_test
    inputs = np.reshape(inputs, (-1, 1))
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (165, 1, 1))
    y_pred = regressor(inputs).detach().to(torch.device('cpu')).numpy()
    y_pred = sc.inverse_transform(y_pred)

    # Visualising Result
    plt.figure
    plt.plot(y_test, color='red', label='Real Web View')
    plt.plot(y_pred, color='blue', label='Predicted Web View')
    plt.title('Web View Forecasting')
    plt.xlabel('Number of Days from Start')
    plt.ylabel('Web View')
    plt.legend()
    plt.show()
    plt.savefig('./out_fig.png')

