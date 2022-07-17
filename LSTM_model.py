
# TODO build a NN
from datetime import date
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data
import math
from torch.utils import data
import scipy.signal


def normalize(data):
    return np.log(data + 1)


def denormalize(data):
    return np.exp(data) - 1


data_raw = load_data.read_data("./data/owid-covid-data.csv", "location", "World")

sequence_length_X = 50
sequence_length_Y = 30
data_cases = np.array(load_data.extract_data(data_raw, ["new_cases_smoothed"])).reshape(
    -1
)

features = np.array(load_data.extract_data(data_raw, ['date'])).reshape(
    -1
)
features = np.array(features, dtype=np.datetime64)
features = np.array(features, dtype=np.int64) - np.array(features, dtype=np.int64)[0]
data_days = np.arange(data_cases.shape[0]).reshape(-1)
data_cases = data_cases[100:]
data_days = data_days[100:]
features = features[100:]

leftover = data_cases.shape[0] % math.lcm(sequence_length_X, sequence_length_Y)
data_cases = data_cases[leftover:]
data_days = data_days[leftover:]
valid_data = ~np.isnan(data_cases)
data_cases = normalize(data_cases)
data_cases = data_cases[valid_data]
data_days = data_days[valid_data]
leftovers = features.shape[0] % math.lcm(sequence_length_X, sequence_length_Y)
features = features[leftovers:]
features = features[valid_data] - features[valid_data][0]

#距离前一个极小值的天数
for i in range(len(features)):
    
    index = scipy.signal.argrelextrema(data_cases[0: i],np.greater)[0]#(data_cases[0: i], distance=120)[0]
    if len(index) > 0:
        features[i] = features[i] - features[index[-1]]
    #print(index)
#print(features)
    

length = data_cases.shape[0] - sequence_length_X - sequence_length_Y

data_days_X_train = data_days[:length]
data_days_X_test = data_days[length:length + sequence_length_Y]

data_cases_X = np.zeros((length, sequence_length_X))
data_cases_Y = np.zeros((length, sequence_length_Y))
data_features_X = np.zeros((length, sequence_length_X))
data_features_Y = np.zeros((length, sequence_length_Y))
for i in range(length):
    data_cases_X[i, :] = data_cases[i : i + sequence_length_X]
    data_cases_Y[i, :] = data_cases[
        i + sequence_length_X : i + sequence_length_X + sequence_length_Y
    ]
    data_features_X[i, :] = features[i : i + sequence_length_X]
    data_features_Y[i, :] = features[
        i + sequence_length_X : i + sequence_length_X + sequence_length_Y
    ]

train_X = torch.tensor(
    np.concatenate((data_cases_X.reshape(-1, sequence_length_X, 1), data_features_X.reshape(-1,sequence_length_X,1)), axis= 2), dtype=torch.float
)

train_Y = torch.tensor(data_cases_Y.reshape(-1, sequence_length_Y), dtype=torch.float)

test_X = torch.tensor(
    np.concatenate((data_cases[-sequence_length_X - sequence_length_Y : -sequence_length_Y].reshape(-1, sequence_length_X, 1), features[-sequence_length_X - sequence_length_Y : -sequence_length_Y].reshape(-1,sequence_length_X,1)), axis= 2), dtype=torch.float
)

test_Y = torch.tensor(
    data_cases[-sequence_length_Y:].reshape(-1, sequence_length_Y), dtype=torch.float
)
#print(train_X.shape)
#print(train_Y.shape)
#print(test_X.shape)
#print(test_Y.shape)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
        )    
        self.linear = nn.Linear(100 * sequence_length_X, sequence_length_Y)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 100 * sequence_length_X)
        x = self.linear(x)
        return x

#把数据和模型转到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LSTM().to(device)
train_X = train_X.to(device)
train_Y = train_Y.to(device)
test_X = test_X.to(device)
test_Y = test_Y.to(device)


# 模型训练
#model = LSTM()
optimzer = torch.optim.Adam(model.parameters(), lr=0.002)
loss_func = nn.MSELoss()
model.train()
l = []
epochs = 10000
for i in range(epochs):
    out = model(train_X)
    loss = loss_func(out, train_Y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    l.append(loss.item())
    if i % 100 == 0:
        print("epoch:", i, "loss:", loss.item())

y_train_predict = model(train_X)
y_test_predict = model(test_X)

plt.figure()
plt.plot(
    data_days_X_train.reshape(-1),
    denormalize(y_train_predict.cpu().detach().numpy()[:, 0].reshape(-1)),
    #denormalize(y_train_predict.detach().numpy()[:, 0].reshape(-1)),
    label="predict train",
    c="b",
)
plt.plot(
    data_days_X_train.reshape(-1),
    denormalize(train_Y.cpu().detach().numpy()[:, 0].reshape(-1)),
    label="true train",
    c="r",
)

plt.plot(
    data_days_X_test.reshape(-1),
    denormalize(y_test_predict.cpu().detach().numpy()[:, :].reshape(-1)),
    label="predict test",
    c="g",
)
plt.plot(
    data_days_X_test.reshape(-1),
    denormalize(test_Y.cpu().detach().numpy()[:, :].reshape(-1)),
    label="true test",
    c="y",
)
plt.legend()
plt.savefig("./result/LSTM_model.png")
