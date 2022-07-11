#TODO build a NN
from pickletools import optimize
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data

data_raw = load_data.read_data('./data/owid-covid-data.csv','location','World')
data = load_data.extract_data(data_raw,['date','new_cases_smoothed'])
length = data.shape[0]
Y = data['new_cases_smoothed']
X = np.arange(length)