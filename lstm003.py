from data import get_data
import keras
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from technical_analysis.generate_labels import Genlabels
from keras.utils import to_categorical
from sklearn.utils import class_weight
import pandas as pd
import os
from bricks_logging.logger import logger
import math
import torch
from torch import nn
from torch.nn import LSTM

dats = {}
minlen = 100000000
bases = ['BTC', 'ETH']
for base in bases:
    dats[base] = get_data(base=base)['odf']
    if len(dats[base]) < minlen:
        minlen = len(dats[base])

for base in bases:
    dats[base] = dats[base][len(dats[base]) - minlen:]

index0 = None
for base in bases:
    if index0 is None:
        index0 = dats[base].index[0]
    else:
        assert index0 == dats[base].index[0]

closes = []
vols = []
df = pd.DataFrame()
columes = []
columedic = {}
BATCH_SIZE = 1
EPOCHS = 20
learning_rate = 0.01
i = 0
for base in bases:
    df[f'{base}_Close0'] = dats[base]['Close']
    df[f'{base}_Close'] = (np.log10(dats[base]['Close']))
    df[f'{base}_Volume'] = dats[base]['Volume']
    columes.append(f'{base}_Close')
    columes.append(f'{base}_Volume')
    columedic[i] = f'{base}_Close'
    i = i + 1
    columedic[i] = f'{base}_Volume'
    i = i + 1
sc_in = MinMaxScaler(feature_range=(0, 1))
sc_df = sc_in.fit_transform(df[columes])
sc_df = pd.DataFrame(sc_df, index=df.index)
sc_df.rename(columns=columedic, inplace=True)


def extract_data(sc_df):
    # obtain labels
    # labels = Genlabels(sc_df[f'BTC_Close'], window=25, polyorder=3, graph=False, smooth=False).labels
    close = np.array(sc_df[f'BTC_Close'])
    dclose = [0]
    dclose2 = np.diff(close, prepend=0)
    for i in range(0, len(close) - 1):
        dclose.append(close[i + 1] - close[i])
        assert dclose2[i + 1] == dclose[-1]
    # labels = np.round((np.power(10, dclose) - 1) * 100)
    # labels[labels < -4] = -4
    # labels[labels > 4] = 4
    dclose = np.array(dclose)
    labels = np.zeros(np.shape(dclose))
    labels[dclose > 0] = 1
    X = []
    for base in bases:
        close = np.array(sc_df[f'{base}_Close'])
        vol = np.array(sc_df[f'{base}_Volume'])
        dclose = np.diff(close, prepend=0)
        X.append(close)
        X.append(vol)
        X.append(dclose)
    X = np.transpose(X)
    y = np.array(labels)
    yindex = sc_df.index
    return X, y, yindex


def shape_data(X, y, yindex, timesteps):
    # scale data
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #
    # if not os.path.exists('models'):
    #     os.mkdir('models')
    #
    # joblib.dump(scaler, 'models/scaler.dump')
    assert len(X) == len(y)
    assert len(y) == len(yindex)

    # reshape data with timesteps
    reshaped = []
    for i in range(timesteps, X.shape[0] + 1):
        reshaped.append(X[i - timesteps:i])

    # account for data lost in reshaping
    X = np.array(reshaped)
    y = y[timesteps - 1:]
    yindex = yindex[timesteps - 1:]
    assert len(X) == len(y)
    assert len(y) == len(yindex)
    return X, y, yindex


def adjust_data(X, y, yindex, test_index=-1, val_num=1):
    # save some data for testing
    test_index = len(y) + test_index
    train_idx = test_index - val_num
    # train_idx = -2
    X_train, y_train, yindex_train = X[:train_idx - 1], y[1:train_idx], yindex[1:train_idx]
    X_val, y_val, yindex_val = X[test_index - 1 - val_num:test_index - 1], y[test_index - val_num:test_index], \
                               yindex[test_index - val_num:test_index]
    x_test, y_test, yindex_test = X[test_index - 1:-1], y[test_index:], yindex[test_index:]
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)
    x_predict = X[-1:]
    return {'train': [X_train, y_train, yindex_train], 'val': [X_val, y_val, yindex_val],
            'test': [x_test, y_test, yindex_test], 'predict': [x_predict]}


nclose = len(sc_df)
oclose = np.array(df['BTC_Close0'])
close_index = df.index
balance = 1
balance1 = 1
balances = []
balances1 = []
balance_close = []
num = 650
predicts = []
dir_name = f'modules2_kdjvol25'
# dir_name = f'modules2_kdjvol16'
os.makedirs(dir_name, exist_ok=True)


class Net(nn.Module):
    def __init__(self, trainx):
        super().__init__()
        self.lstm = LSTM(input_size=trainx.shape[1:])

    def forward(self, input):
        output = input + 1
        return output
