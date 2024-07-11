import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
from datetime import datetime, timedelta

# Function to fetch cryptocurrency data
def fetch_crypto_data(symbol, start, end, interval):
    data = yf.download(symbol, start=start, end=end, interval=interval)
    return data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create datasets
def create_datasets(data, n_past, n_future):
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, 0])
        y.append(data[i + n_future - 1:i + n_future, 0])
    X, y = np.array(X), np.array(y)

    if X.shape[0] > 0 and X.shape[1] > 0:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    else:
        raise ValueError("Insufficient data to reshape X")
    
    return X, y

# Function to build LSTM model
def build_model(input_shape, output_units):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=output_units))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Intervals to train
intervals = ['1d', '1wk']

# Cryptocurrencies to train on
cryptocurrencies = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']

# Get today's date
today = datetime.today().date()
three_years_ago = today - timedelta(days=3*365)

for cryptocurrency in cryptocurrencies:
    for interval in intervals:
        print(f"Training model for {cryptocurrency} with interval {interval}")
        
        # Fetch data
        data_df = fetch_crypto_data(cryptocurrency, start=three_years_ago, end=today, interval=interval)

        if data_df.empty:
            print(f"No data fetched for {cryptocurrency} with interval {interval}")
            continue

        # Preprocess data
        scaled_data, scaler = preprocess_data(data_df)
        n_past = 60
        n_future = 1
        X, y = create_datasets(scaled_data, n_past, n_future)

        if len(X) == 0 or len(y) == 0:
            print(f"Not enough data to create datasets for {cryptocurrency} with interval {interval}")
            continue

        # Build and train the model
        model = build_model((n_past, 1), n_future)
        model.fit(X, y, epochs=50, batch_size=1, validation_split=0.2)

        # Save the model
        model.save(f'data/lstm_model_{cryptocurrency}_{interval}.keras')










"""
latest code>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
from datetime import datetime, timedelta

# Function to fetch cryptocurrency data
def fetch_crypto_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create datasets
def create_datasets(data, n_past, n_future):
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, 0])
        y.append(data[i + n_future - 1:i + n_future, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

# Function to build LSTM model
def build_model(input_shape, output_units):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=output_units))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Get today's date
today = datetime.today().date()
one_year_ago = today - timedelta(days=365)

# Fetch data
data_df = fetch_crypto_data('BTC-USD', start=one_year_ago, end=today)

# Preprocess data
scaled_data, scaler = preprocess_data(data_df)
n_past = 60
n_future = 1
X, y = create_datasets(scaled_data, n_past, n_future)

# Build and train the model
model = build_model((n_past, 1), n_future)
model.fit(X, y, epochs=50, batch_size=1, validation_split=0.2)

# Save the model
model.save('data/lstm_model.keras')
"""


# import yfinance as yf
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# import datetime

# # Function to fetch cryptocurrency data for the last year
# def fetch_crypto_data(symbol):
#     end_date = datetime.datetime.now()
#     start_date = end_date - datetime.timedelta(days=365)
#     data = yf.download(symbol, start=start_date, end=end_date)
#     return data

# # Function to preprocess data
# def preprocess_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
#     return scaled_data, scaler

# # Function to create datasets
# def create_datasets(data, n_past, n_future):
#     X, y = [], []
#     for i in range(n_past, len(data) - n_future + 1):
#         X.append(data[i - n_past:i, 0])
#         y.append(data[i + n_future - 1, 0])
#     X, y = np.array(X), np.array(y)
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     return X, y

# # Fetch data
# cryptocurrency = 'BTC-USD'
# data_df = fetch_crypto_data(cryptocurrency)

# # Preprocess data
# scaled_data, scaler = preprocess_data(data_df)
# n_past = 60
# n_future = 1
# X, y = create_datasets(scaled_data, n_past, n_future)

# # Build LSTM model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(n_past, 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(X, y, epochs=25, batch_size=32)

# # Save the model
# model.save('data/saved_model.keras')

# print("Model training complete and saved.")

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import yfinance as yf
# from datetime import datetime, timedelta

# # Function to fetch cryptocurrency data
# def fetch_crypto_data(symbol, start, end):
#     data = yf.download(symbol, start=start, end=end)
#     return data

# # Function to preprocess data
# def preprocess_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
#     return scaled_data, scaler

# # Function to create datasets
# def create_datasets(data, n_past, n_future):
#     X, y = [], []
#     for i in range(n_past, len(data) - n_future + 1):
#         X.append(data[i - n_past:i, 0])
#         y.append(data[i + n_future - 1:i + n_future, 0])
#     X, y = np.array(X), np.array(y)
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     return X, y

# # Function to build LSTM model
# def build_model(input_shape, output_units):
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
#     model.add(LSTM(units=50))
#     model.add(Dense(units=output_units))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Set the cryptocurrency symbol
# symbol = 'BTC-USD'

# # Define the time range (1 year ago to today)
# end_date = datetime.now()
# start_date = end_date - timedelta(days=365)

# # Fetch data
# data_df = fetch_crypto_data(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# # Preprocess data
# scaled_data, scaler = preprocess_data(data_df)
# n_past = 60
# n_future = 1
# X, y = create_datasets(scaled_data, n_past, n_future)

# # Build and train the model
# model = build_model((n_past, 1), n_future)
# model.fit(X, y, batch_size=1, epochs=50, validation_split=0.2)

# # Save the model
# model.save('data/saved_model.keras')

# import yfinance as yf
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from datetime import datetime, timedelta

# # Fetch cryptocurrency data
# def fetch_crypto_data(symbol, start, end):
#     data = yf.download(symbol, start=start, end=end)
#     return data

# # Preprocess data
# def preprocess_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
#     return scaled_data, scaler

# # Create datasets
# def create_datasets(data, n_past, n_future):
#     X, y = [], []
#     for i in range(n_past, len(data) - n_future + 1):
#         X.append(data[i - n_past:i, 0])
#         y.append(data[i + n_future - 1:i + n_future, 0])
#     X, y = np.array(X), np.array(y)
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     return X, y

# # Build LSTM model
# def build_model(input_shape, output_units):
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
#     model.add(LSTM(units=50))
#     model.add(Dense(units=output_units))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Define the time range (1 year ago to today)
# end_date = datetime.now()
# start_date = end_date - timedelta(days=365)
# # Fetch data
# # data_df = fetch_crypto_data('BTC-USD', '2011-01-01', '2024-05-23')
# data_df = fetch_crypto_data('BTC-USD', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# # Preprocess data
# scaled_data, scaler = preprocess_data(data_df)
# n_past = 60
# n_future = 1
# X, y = create_datasets(scaled_data, n_past, n_future)

# # Split data
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build and train the model
# model = build_model((n_past, 1), n_future)
# model.fit(X_train, y_train, batch_size=1, epochs=30, validation_split=0.2)

# # Save the model in the .keras format
# model.save('data/saved_model.keras')



