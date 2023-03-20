from data import get_data
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dropout, Dense, BatchNormalization
# from keras.layers import CuDNNLSTM as LSTM
from keras.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from technical_analysis.generate_labels import Genlabels
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.utils import class_weight
import pandas as pd
import os
from bricks_logging.logger import logger
import math

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
for j in range(num, 0, -1):
    curr_df = sc_df[1000 + num - j:nclose - j]
    X, y, yindex = extract_data(curr_df)
    X, y, yindex = shape_data(X, y, yindex, 2)
    data = adjust_data(X, y, yindex)  # 回测
    # data = adjust_data(X, y, yindex, train_idx=-1, test_index=0)  # 预测当天涨跌
    yi = f'{yindex[-1]}'[:-9]
    model_file_name = f'{dir_name}/{yi}.h5'
    if os.path.exists(model_file_name):
        model = keras.models.load_model(model_file_name)
    else:
        train_x = data['train'][0]
        train_y = data['train'][1]
        validation_x, validation_y, _ = data['val']
        model = Sequential()
        model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, train_x.shape[1], train_x.shape[2]),
                       input_shape=(train_x.shape[1:]),
                       return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        # model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, train_x.shape[1], train_x.shape[2]), return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())

        model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, train_x.shape[1], train_x.shape[2]), ))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='softmax'))

        opt = Adam(learning_rate=learning_rate, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        # # Compile model
        # model.compile(
        #     loss='sparse_categorical_crossentropy',
        #     optimizer=opt,
        #     metrics=['sparse_categorical_accuracy']
        # )
        #
        # tensorboard = TensorBoard(log_dir="logs/{}".format(dir_name))
        #
        # NNNAME = "LSTM_STATEFUL"
        # # unique file name that will include the epoch and the validation acc for that epoch
        # filepath = NNNAME + "-{epoch:02d}-{val_acc:.3f}"
        # checkpoint = ModelCheckpoint("models/{}.model".format(
        #     filepath, monitor='val_sparse_categorical_accuracy',
        #     verbose=1, save_best_only=True,
        #     mode='max'
        # ))  # saves only the best ones
        # Train model
        history = model.fit(
            train_x, train_y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            shuffle=False,
            validation_data=(validation_x, validation_y),
            # class_weight=dict(enumerate(train_weights)),
            # callbacks=[tensorboard, checkpoint],
            # class_weight=train_weights,
        )
        # val_weights = class_weight.compute_class_weight('balanced',
        #                                                 np.unique(validation_y),
        #                                                 validation_y)
        #
        # val_sample_weights = training_utils.standardize_weights(np.array(validation_y),
        #                                                         class_weight=dict(enumerate(val_weights)))

        # Score model
        # score = model.evaluate(validation_x, validation_y, verbose=0, sample_weight=val_sample_weights)
        # score = model.evaluate(validation_x, validation_y, verbose=0)
        model.save(model_file_name)
    if data['test'][0].size > 0:  # 存在时回测
        y_predict = model.predict(data['test'][0])
        y_predict_val = model.predict(data['val'][0])
        close_today = oclose[nclose - 2 - j]
        close_Next = oclose[nclose - 1 - j]  # 预测倒数第几天
        assert close_index[nclose - 1 - j] == curr_df.index[-1]
        assert data['val'][2][-1] == close_index[nclose - 2 - j]
        # assert data['train'][2][-1] == close_index[nclose - 3 - j]
        btc_close = np.array(curr_df['BTC_Close'])
        # assert data['train'][0][-1][-1][0] == btc_close[-4]
        assert data['val'][0][-1][-1][0] == btc_close[-3]
        balance_close.append(close_Next)
        assert len(y_predict[-1]) == 2
        # if np.sum(y_predict[-1][0:4]) < np.sum(y_predict[-1][5:]):  # 看涨
        if y_predict[-1][0] < 0.5:  # 看涨
            balance = balance * close_Next / close_today
            balance1 = balance1 * np.power(10, btc_close[-1] - btc_close[-2])
        else:
            balance = balance * 2 - balance * close_Next / close_today
            balance1 = balance1 * 2 - balance1 * np.power(10, btc_close[-1] - btc_close[-2])
        balances.append(balance)
        balances1.append(balance1)
        predicts.append(y_predict[-1][0])
        unknow_predict = model.predict(data['predict'][0])
        logger.info(
            f'{j},{unknow_predict}, {yindex[-1]}, b:{balance}, b1:{balance1}: p:{y_predict[-1]},pv:{y_predict_val[-1]},t:{close_today}, n:{close_Next}, v:{data["val"][1]}')
    else:  # 预测今天收盘涨跌
        unknow_predict = model.predict(data['predict'][0])
        logger.info(f'predict:{unknow_predict}')
plt.plot(balances, label='balance')
plt.plot(balances1, label='balance1')
plt.plot(np.array(balance_close) / balance_close[0], label='close')
plt.plot(predicts, label='predicts')
plt.legend()
plt.savefig(f'{dir_name}.jpg')
plt.show()
