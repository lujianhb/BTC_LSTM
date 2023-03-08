import os
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def get_data():
    file_name = 'btc_usdt.csv'
    quote = 'USDT'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
        # param = {"convert": "USDT", "slug": "bitcoin", "time_end": "1601510400", "time_start": "1367107200"}
        param = {"convert": quote, "slug": "bitcoin", "time_end": "1677422213", "time_start": "1367107200"}
        content = requests.get(url=url, params=param).json()
        df = pd.json_normalize(content['data']['quotes'])
        df.to_csv(file_name, index=False)
    # Extracting and renaming the important variables
    df['Date'] = pd.to_datetime(df[f'quote.{quote}.timestamp']).dt.tz_localize(None)
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
    df['Next'] = df['Close'].shift(-1)
    df = df.dropna()
    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Next']
    sc_df = pd.DataFrame()
    for col in columns:
        sc_df[col] = np.log10(df[col])
    # normalizing the exogeneous variables
    sc_in = MinMaxScaler(feature_range=(0, 1))
    sc_df = sc_in.fit_transform(sc_df[columns])
    sc_df = pd.DataFrame(sc_df, index=df.index)
    sc_df.rename(columns={0: 'Open', 1: 'High', 2: 'Low', 3: 'Close', 4: 'Volume', 5: 'Next'}, inplace=True)
    return {
        'trans': sc_in,
        'df': sc_df,
        'odf': df,
    }
