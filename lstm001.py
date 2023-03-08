from data import get_data
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from matplotlib import pyplot as plt
import numpy as np
from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from keras.utils import to_categorical
import os
from bricks_logging.logger import logger

dat = get_data()

df = dat['df']
odf = dat['odf']
trans = dat['trans']


def extract_data(data, vol):
    # obtain labels
    labels = Genlabels(data, window=25, polyorder=3, graph=False, smooth=False).labels

    # obtain features
    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    # dpo = Dpo(data, period=4).values
    # cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    # inter_slope = PolyInter(data, progress_bar=True).values

    # truncate bad values and shift label
    # X = np.array([macd[30:-1],
    #               stoch_rsi[30:-1],
    #               inter_slope[30:-1],
    #               dpo[30:-1],
    #               cop[30:-1]])
    data0 = list(data)
    data0.insert(0, data[0])
    ddata = np.diff(data0)
    X = np.array([
       # macd[30:-1],
       # stoch_rsi[30:-1],
        data[30:-1],
        ddata[30:-1],
        vol[30:-1],
    ])
    X = np.transpose(X)
    labels = labels[31:]
    return X, labels


def shape_data(X, y, timesteps):
    # scale data
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #
    # if not os.path.exists('models'):
    #     os.mkdir('models')
    #
    # joblib.dump(scaler, 'models/scaler.dump')

    # reshape data with timesteps
    reshaped = []
    for i in range(timesteps, X.shape[0] + 1):
        reshaped.append(X[i - timesteps:i])

    # account for data lost in reshaping
    X = np.array(reshaped)
    y = y[timesteps - 1:]

    return X, y


def adjust_data(X, y):
    # save some data for testing
    train_idx = -2
    validate_idx = -1
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:validate_idx], y[train_idx:validate_idx]
    x_test, y_test = X[validate_idx:], y[validate_idx:]
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)
    return {'train': [X_train, y_train], 'val': [X_val, y_val], 'test': [x_test, y_test]}


close = np.array(df['Close'])
oclose = np.array(odf['Close'])
assert len(close) == len(oclose)
vol = np.array(df['Volume'])
nclose = len(close)
balance = 1
balances = []
balance_close = []
num = 650
predicts = []
# dir_name = f'modules2_kdjvol4_nosmoot10'
dir_name = f'modules2_kdjvol6'
os.makedirs(dir_name, exist_ok=True)
for j in range(num, 350, -1):
    if j == 645:
        debug = 1
    X, y = extract_data(close[:nclose - j], vol[:nclose - j])
    X, y = shape_data(X, y, 10)
    data = adjust_data(X, y)
    model_file_name = f'{dir_name}/lstm_model_{j}.h5'
    if os.path.exists(model_file_name):
        model = keras.models.load_model(model_file_name)
    else:
        model = Sequential()
        model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))

        # second layer
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))

        # fourth layer and output
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        # compile layers
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(
            data['train'][0], data['train'][1],
            epochs=100, batch_size=8, shuffle=True,
            validation_data=(data['val'][0], data['val'][1])
        )
        model.save(model_file_name)
    y_predict = model.predict(data['test'][0])
    y_predict_val = model.predict(data['val'][0])
    close_today = oclose[nclose - 2 - j]
    close_Next = oclose[nclose - 1 - j]  # 预测倒数第几天
    balance_close.append(close_Next)
    if y_predict[-1][0] < 0.5:  # 看涨
        balance = balance * close_Next / close_today
    elif y_predict[-1][0] > 0.5:
        balance = balance * 2 - balance * close_Next / close_today
    balances.append(balance)
    predicts.append(y_predict[-1][0])
    # assert close[nclose - 4 - j] == data['train'][0][-1][-1][
    #     2], f'{close[nclose - 4 - j]}, {data["train"][0][-1][-1][2]}'
    # assert close[nclose - 3 - j] == data['val'][0][-1][-1][2], f'{close[nclose - 3 - j]}, {data["val"][0][-1][-1][2]}'
    # assert close[nclose - 2 - j] == data['test'][0][-1][-1][2], f'{close[nclose - 2 - j]}, {data["test"][0][-1][-1][2]}'
    logger.info(
        f'{j}, b:{balance}, p:{y_predict[-1][0]},pv:{y_predict_val[-1][0]},t:{close_today}, n:{close_Next}, v:{data["val"][1][-1, :]}')
plt.plot(balances, label='balance')
plt.plot(np.array(balance_close) / balance_close[0], label='close')
plt.plot(predicts, label='predicts')
plt.legend()
plt.savefig(f'{dir_name}.jpg')
plt.show()
