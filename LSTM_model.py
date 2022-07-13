# TODO build a NN
from operator import length_hint
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data

def normalize(data):
    return np.log(data + 1)

def denormalize(data):
    return  np.exp(data) - 1

data_raw = load_data.read_data("./data/owid-covid-data.csv",'location','World')

sequence_length = 3
data_cases = np.array(load_data.extract_data(data_raw,['new_cases_smoothed'])).reshape(-1)
data_days = np.arange(data_cases.shape[0]).reshape(-1)
data_cases = data_cases[100:]
data_days = data_days[100:]

leftover = data_cases.shape[0] % sequence_length
data_cases = data_cases[leftover:]
data_days = data_days[leftover:]
valid_data = ~np.isnan(data_cases)
data_cases = normalize(data_cases)
data_cases = data_cases[valid_data]
data_days = data_days[valid_data]


length = data_cases.shape[0] - sequence_length

data_days_X = data_days[:length]

data_cases_X = np.zeros((length,sequence_length))
data_cases_Y = np.zeros((length,1))
for i in range(length):
    data_cases_X[i,:] = data_cases[i:i+sequence_length]
    data_cases_Y[i,:] = data_cases[i+sequence_length]



train_X = torch.tensor(data_cases_X.reshape(-1,sequence_length,1), dtype=torch.float)
train_Y = torch.tensor(data_cases_Y.reshape(-1,1,1), dtype=torch.float)



class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=1, batch_first=True)
        self.linear = nn.Linear(100 * sequence_length, 1)
    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 100 * sequence_length)
        x = self.linear(x)
        return x

# 模型训练
model = LSTM()
optimzer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_func = nn.MSELoss()
model.train()
l = []
epochs = 1000
for i in range(epochs):
    output = model(train_X)
    loss = loss_func(output, train_Y)
    l.append(loss)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if i % 100 == 0:       
        print("i:{}, train_loss:{}".format(i, loss))


y_predict = model(train_X)

plt.figure()
plt.plot(data_days_X.reshape(-1) ,denormalize(y_predict.detach().numpy().reshape(-1)),label='predict')
plt.show()

