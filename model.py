# TODO build a NN
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data

data_raw = load_data.read_data("./data/owid-covid-data.csv",'location','United States')
# data = load_data.extract_data(
#     data_raw,
#     [
#         'reproduction_rate',
#         'total_vaccinations',
#         'people_vaccinated',
#         'people_fully_vaccinated',
#         'total_boosters',
#         'new_vaccinations',
#         'new_vaccinations_smoothed',
#         'total_vaccinations_per_hundred',
#         'people_vaccinated_per_hundred',
#         'people_fully_vaccinated_per_hundred',
#         'total_boosters_per_hundred',
#         'new_vaccinations_smoothed_per_million',
#         'new_people_vaccinated_smoothed',
#         'new_people_vaccinated_smoothed_per_hundred',
#         'stringency_index',
#         'population',
#         'population_density',
#         'median_age',
#         'aged_65_older',
#         'aged_70_older',
#         'gdp_per_capita',
#         'extreme_poverty',
#         'cardiovasc_death_rate',
#         'diabetes_prevalence',
#         'female_smokers',
#         'male_smokers',
#         'handwashing_facilities',
#         'hospital_beds_per_thousand',
#         'life_expectancy',
#         'human_development_index',
#     ],
# )
data_cases = np.array(load_data.extract_data(data_raw,['new_cases_smoothed']))
data_days = np.arange(data_cases.shape[0]).reshape(-1,1)
valid_data = ~np.isnan(data_cases)
data_cases = data_cases[valid_data][100:]
data_days = data_days[valid_data][100:]

tensor_X = torch.tensor(data_days, dtype=torch.float32,requires_grad=True).reshape(-1,1,1)
tensor_Y = torch.tensor(data_cases, dtype=torch.float32).reshape(-1,1)
class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Lstm, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, _ = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out , _

model = Lstm(1, 100, 1)
loss_fuction = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1)

epochs = 10000
hidden =  (torch.zeros(1, 1, 100), torch.zeros(1, 1, 100))

for i in range(epochs):
    hidden =  (torch.zeros(1 , 1 , 100), torch.zeros(1, 1, 100))
    y_hat,hidden = model(tensor_X , hidden)
    loss = loss_fuction(y_hat, tensor_Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"epoch {i} loss {loss.item()}")

hidden =  (torch.zeros(1, 1, 100), torch.zeros(1, 1, 100))
X_future = torch.arange(100,data_days.shape[0] + 100,dtype=torch.float32).reshape(-1,1,1)
Y_future,_ = model(X_future,hidden)

plt.figure()
plt.plot(X_future.squeeze().detach().numpy(),Y_future.squeeze().detach().numpy())
plt.show()

