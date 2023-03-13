import os
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import datetime

slugs = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
}


def get_data(base='BTC', quote='USDT', data_dir='data1'):
    assert str.isupper(base)
    assert str.isupper(quote)
    os.makedirs(data_dir, exist_ok=True)
    file_name = f'{base.lower()}_{quote.lower()}.csv'
    file_path = os.path.join(data_dir, file_name)
    slug = slugs[base]
    nowt = datetime.datetime.now().timestamp()
    nowt = int(nowt)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
        param = {"convert": quote, "slug": slug, "time_end": f'{nowt}', "time_start": "1367107200"}
        content = requests.get(url=url, params=param).json()
        df = pd.json_normalize(content['data']['quotes'])
        df.to_csv(file_path, index=False)
    # Extracting and renaming the important variables
    # df['Date'] = pd.to_datetime(df[f'quote.{quote}.timestamp']).dt.tz_localize(None)
    df['Date'] = pd.to_datetime(df[f'time_open']).dt.tz_localize(None)
    df['Low'] = df[f'quote.{quote}.low']
    df['High'] = df[f'quote.{quote}.high']
    df['Open'] = df[f'quote.{quote}.open']
    df['Close'] = df[f'quote.{quote}.close']
    df['Volume'] = df[f'quote.{quote}.volume']

    # Drop original and redundant columns
    df = df.drop(
        columns=['time_open', 'time_close', 'time_high', 'time_low', f'quote.{quote}.low', f'quote.{quote}.high',
                 f'quote.{quote}.open', f'quote.{quote}.close', f'quote.{quote}.volume', f'quote.{quote}.market_cap',
                 f'quote.{quote}.timestamp'])
    # Creating a new feature for better representing day-wise values
    # df['Mean'] = (df['Low'] + df['High']) / 2
    # Cleaning the data for any NaN or Null fields
    df = df.dropna()
    # date time typecast
    df['Date'] = pd.to_datetime(df['Date'])
    df.index = df['Date']
    # df['Next'] = df['Close'].shift(-1)
    # df = df.dropna()
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    sc_df = pd.DataFrame()
    for col in columns:
        sc_df[col] = np.log10(df[col])
    # normalizing the exogeneous variables
    sc_in = MinMaxScaler(feature_range=(0, 1))
    sc_df = sc_in.fit_transform(sc_df[columns])
    sc_df = pd.DataFrame(sc_df, index=df.index)
    sc_df.rename(columns={0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume'}, inplace=True)
    return {
        'trans': sc_in,
        'df': sc_df,
        'odf': df,
    }
