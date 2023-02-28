#!/usr/bin/env python
# coding: utf-8

# #                                  比特币交易策略初试（多因子LSTM最简模型）

# ## 比特币涨跌原因总结：
# 
# ### 1.NVT Ratio，即Network Value to Transactions Ratio的缩写
#      NVT比率是目前看到的认可度较高的比特币估值模型。
#      NVT比率（网络价值与交易比率）类似于股票市场中使用的PE比率。
#      当比特币的NVT高时，表明其网络估值超过其支付网络上传输的价值，可能为一个不可持续的泡沫。
# ### 2.比特币关注度：政策，社会动荡（避险），舆论
# ### 3.比特币开采难度：矿机价格，算力，电费，剩余可开采币（总量有限为2100万个）
# ### 4.每时刻价格受历史价格影响
# ### 5.总量少，对于大宗持有者来说，退出进来都会改变这个环境，难以在波动中获利
# ### 6.每日交易笔数限制：区块大小、分叉等
# ### 7.交易所bug
# ### 8.其它因子：https://data.bitcoinity.org/bitcoin/block_size_votes/7d?c=block_size_votes&t=b
# ***

# ## 模型思路： 
# 1.预测价格：输入过去10天各因子序列（本项目因子包括比特币每天价格，每天交易量，每天NVT比率），输出将来2天的比特币收盘价。
# 
# 2.买卖信号：资产分2份，2天为一周期，每天都判断，如2天后的收盘价大于当前价则买一份，买信号发出2天后必卖一份，策略偏保守，最大回撤小，收益小。
# 
# 3.仓位： 根据信号计算应该持有仓位，要加仓则买，要减仓则卖。
# 
# 4.回测：不考虑滑点、手续费、假设环境不会因自己操作变化，计算资金曲线，计算最大回撤；
# ***

# ## 读取比特币历史数据
# 
# 数据来源：https://github.com/yan-wong/BitcoinPriceHistoryInChina/blob/master/data/okcoin/daily_price_btc_cny.csv
# 

# In[2]:


import sys
import pandas as pd

df = pd.read_csv("./daily_price_btc_cny.csv", sep=',')
header = ['date', 'open', 'high', 'low', 'close', 'volume']
df = df[header]
df['date'] = pd.to_datetime(df['date'])
df.set_index(['date'], inplace=True, drop=True)
df.head()

# ## 读取比特币NVT Ratio
# 
# 数据来源：https://docs.google.com/spreadsheets/d/1xLTC4oaDyI-aqbc_lehGUspGDRLb-q6s7SN4nTdGluY/edit#gid=1383584969
# 
# NVT介绍：https://woobull.com/introducing-nvt-ratio-bitcoins-pe-ratio-use-it-to-detect-bubbles/

# In[3]:


import sys
import pandas as pd

df_nvt = pd.read_csv("./btc_nvt.csv", sep=',')
df_nvt.set_index(['date'], inplace=True, drop=False)
df_nvt = df_nvt.loc['9/1/2013': '9/18/2017']
df_nvt['date'] = pd.to_datetime(df_nvt['date'])
df_nvt.set_index(['date'], inplace=True, drop=True)
df_nvt = df_nvt._convert(numeric=True)
header = ['Network Value', 'Txn Value', 'NVT']
df_nvt = df_nvt[header]
n_ma = 5
df_nvt['ma_' + str(n_ma)] = df_nvt['NVT'].rolling(n_ma, min_periods=1).mean()
# print(df_nvt.dtypes)
df_nvt.head()

# In[60]:


import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')
# print(df_nvt.values, df_nvt.values[:,2].shape)
plt.figure()
plt.subplot(5, 1, 1)
plt.title('Network Value')
plt.plot(df_nvt.index, df_nvt.values[:, 0].reshape(-1))
plt.subplot(5, 1, 3)
plt.title('Txn Value')
plt.plot(df_nvt.index, df_nvt.values[:, 1].reshape(-1))
plt.subplot(5, 1, 5)
plt.title('NVT')
plt.plot(df_nvt.index, df_nvt.values[:, 2].reshape(-1))
plt.plot(df_nvt.index, df_nvt.values[:, 3].reshape(-1))
plt.show()

# ## 合并数据

# In[61]:


df['NVT'] = df_nvt['NVT']
df['NVT'] = df['NVT'].fillna(method='ffill')
df.to_csv('input.csv')
df.head()

# In[62]:


import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')
df = read_csv('input.csv')
df.set_index(['date'], inplace=True, drop=True)
list_feature = list(df.columns)
print(list_feature)
print(df.head())
print(df.tail())

# In[63]:


# plt.figure()
# for n_col, str_col in enumerate(df.columns):
#     plt.subplot(len(df.columns)*3-1,1,1+(n_col*3))
#     plt.title(str_col, y=0.5, loc='right')
#     plt.plot(df.index, df.values[:,n_col].reshape(-1))
# plt.show()


# ## 多因子LSTM预测模型

# ### 特征归一化

# In[64]:


from sklearn.preprocessing import MinMaxScaler

# ensure all data is float
values = df
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print(scaled.shape)
scaled

# ### 将数据集转换为监督学习问题

# In[65]:


from pandas import DataFrame
from pandas import concat


def series_to_supervised(data, list_feature, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(list_feature[j] + '(t-%d)' % (i)) for j in range(n_vars)]  # 待优化
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(list_feature[j] + '(t)') for j in range(n_vars)]
        else:
            names += [(list_feature[j] + '(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


n_in = 3  # 用过去多少天的特征作为输入
n_out = 2  # 预测未来多少天价格
# frame as supervised learning
reframed = series_to_supervised(scaled, list_feature, n_in, n_out)
print(reframed.shape)
reframed.head()

# ### 拆分为输入矩阵和输出矩阵

# In[66]:


print(reframed.columns)
list_use_feature = []
for j in range(len(list_feature)):
    #     list_use_feature.append(list_feature[j]+'(t)')
    for i in range(n_in):
        list_use_feature.append(list_feature[j] + '(t-%d)' % (i + 1))

X = reframed[list_use_feature]
print(X.columns)

list_use_result = []
list_use_result.append('close(t)')
for i in range(n_out - 1):
    list_use_result.append('close(t+%d)' % (i + 1))
Y = reframed[list_use_result]
print(Y.columns)

# ### 拆分为训练集和测试集（最好有验证集来网格搜素或贝叶斯搜素超参数，这里采用最简单方式）

# In[67]:


print(X.shape, Y.shape)
type(X)

# In[68]:


X = X.values
Y = Y.values
n_train_days = int(X.shape[0] / 3)
print(n_train_days)
train_X, train_Y = X[:n_train_days, :], Y[:n_train_days, :]
test_X, test_Y = X[n_train_days:, :], Y[n_train_days:, :]

# reshape input to be 3D [batchs, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# train_Y = train_Y.reshape((train_Y.shape[0], 1, train_Y.shape[1]))
# test_Y = test_Y.reshape((test_Y.shape[0], 1, test_Y.shape[1]))
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

# ### 构建及训练LSTM模型

# In[69]:


from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(train_Y.shape[1]))
# model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_Y, epochs=50, batch_size=72, validation_data=(test_X, test_Y), verbose=2,
                    shuffle=False)
# history = model.fit(train_X, train_Y[:,:,0], epochs=50, batch_size=72, validation_data=(test_X, test_Y[:,:,0]), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# In[70]:


Yhat = model.predict(test_X)

# ### 评估模型
# 注意：close(t)[i+1] 本就不应该等于 close(t+1)[i]， 因变量不同 
# 
# 预测未来值较多时，应该改为for循环

# In[71]:


from numpy import concatenate
import copy

# make a prediction
inv_Yhat = copy.deepcopy(Yhat)
print('close(t)', inv_Yhat[:, 0].reshape(-1)[:10])
print('close(t+1)', inv_Yhat[:, 1].reshape(-1)[:10])
print(Yhat.shape)
print(scaled[n_train_days:, :].shape)
inv_Yhat0 = concatenate((scaled[n_in:-n_out + 1, :3][n_train_days:, :],
                         Yhat[:, :1],
                         scaled[n_in:-n_out + 1, 4:][n_train_days:, :]), axis=1)
inv_Yhat0 = scaler.inverse_transform(inv_Yhat0)
inv_Yhat0 = inv_Yhat0[:, 3]

inv_Yhat1 = concatenate((scaled[n_in:-n_out + 1, :3][n_train_days:, :],
                         Yhat[:, 1:],
                         scaled[n_in:-n_out + 1, 4:][n_train_days:, :]), axis=1)
inv_Yhat1 = scaler.inverse_transform(inv_Yhat1)
inv_Yhat1 = inv_Yhat1[:, 3]

predictiony0 = inv_Yhat0.reshape(-1)
predictiony1 = inv_Yhat1.reshape(-1)
originaly = values['close'].values[n_in:-n_out + 1][n_train_days:]
x = values.index[n_in:-n_out + 1][n_train_days:]
print(len(predictiony0), predictiony0[:10])
print(len(predictiony1), predictiony1[:10])
print(len(originaly), originaly[:10])
print(len(x), x[:10])
print(values.index)

# ### 预测基本准确

# In[72]:


# get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
# plt.subplot(5,1,1)
# plt.plot(x, predictiony0)
plt.plot(predictiony0)
# plt.subplot(5,1,3)
plt.plot(predictiony1)
# plt.subplot(5,1,5)
plt.plot(originaly)
plt.show()

# ### 计算信号 (只计算测试集)
# 资产分2份，2天为一周期，每天都判断，如2天后的收盘价大于当前价则买一份，买信号发出2天后必卖一份，策略偏保守，最大回撤小，收益小。

# In[73]:


dataset = {'date': x,
           'close(t-1)': originaly,
           'close(t)': predictiony0,
           'close(t+1)': predictiony1}
df = DataFrame(dataset)
# df.set_index(['date'], inplace = True, drop=True) 
df['close(t+1)-close(t-1)'] = df['close(t+1)'] - df['close(t-1)']
df['isbuy'] = df['close(t+1)-close(t-1)'] > 0
df['issell'] = df['isbuy'].shift(2)  # n_out=2
df['issell'].fillna(value=False, inplace=True)
df

# ### 计算仓位

# In[74]:


initlocation = 0  # 初始仓位为0
initUS = 2  # 初始美元现金为2
maxlocation = 2  # 满仓为2


def function(isbuy, issell):
    if issell and (not isbuy): return -1
    if isbuy and issell: return 0
    if (not isbuy) and (not issell): return 0
    if isbuy and (not issell): return 1


df['location action'] = df.apply(lambda x: function(x.isbuy, x.issell), axis=1)
df['location status'] = df['location action'].expanding().sum()
df['location status'] = df['location status'] / df['location status'].max()
df

# ### 计算资金曲线

# In[75]:


df['cycle growth rate'] = df['close(t-1)'].rolling(2).apply(lambda x: x[1] - x[0])
df['cycle growth rate'] = df['cycle growth rate'].shift(-1)
df['cycle growth rate'].fillna(value=0, inplace=True)
df['cycle growth rate'] = df['cycle growth rate'] / df['close(t-1)']  # close(t-1)为真实价格，close(t)，close(t+1)为预测价格
df['Asset cycle growth rate'] = df['location status'] * df['cycle growth rate']
df['cycle growth rate'] = df['cycle growth rate'].shift(1)
df['cycle growth rate'].fillna(value=0, inplace=True)
df['Asset cycle growth rate'] = df['Asset cycle growth rate'].shift(1)
df['Asset cycle growth rate'].fillna(value=0, inplace=True)
df['Asset cycle growth rate'] = df['Asset cycle growth rate'] + 1
df['Asset growth rate'] = df['Asset cycle growth rate'].cumprod()
df['BTC growth rate'] = (df['cycle growth rate'] + 1).cumprod()
df

# ### 绘制资金曲线

# In[76]:


# get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
# plt.subplot(3,1,1)
plt.plot(df['BTC growth rate'].values.reshape(-1), 'r', label='BTC growth rate')
plt.plot(df['Asset growth rate'].values.reshape(-1), 'b', label='Asset growth rate')

plt.legend()
plt.show()
# print(df['date'].head())
# print(df['date'].tail())


# ## 总结：
# ### 此策略会减少最大回撤，减少波动，比较稳健；
# ### 收益率与LSTM预测的准确率直接相关，应该加入更多因子，及设置更合适的超参
# ### 此模型同样的数据收益率存在随机性
# ### 试用集成学习等模型

# In[ ]:
