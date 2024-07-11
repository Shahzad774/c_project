import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from PIL import Image

def fetch_crypto_data(symbol, start, end, interval):
    data = yf.download(symbol, start=start, end=end, interval=interval)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

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

st.set_page_config(page_title='Cryptocurrency Price Prediction', layout='wide')

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Prediction"],
        icons=["house", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
    )


if selected == "Home":
    st.title('Cryptocurrency Price Prediction')
    st.image('crypto.jpg', use_column_width=True) 
    st.write("""
    This app predicts the future prices of cryptocurrencies using a Long Short-Term Memory (LSTM) model.
    Select a cryptocurrency from the sidebar to get started.
    """)

if selected == "Prediction":
    st.title('Cryptocurrency Price Prediction')

    cryptocurrency = st.selectbox('Select Cryptocurrency', ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD'])
    interval = st.selectbox('Select Interval', ['1d', '1wk'])

    today = datetime.today().date()
    three_years_ago = today - timedelta(days=3*365)

    data_df = fetch_crypto_data(cryptocurrency, start=three_years_ago, end=today, interval=interval)
    
    st.subheader(f'{cryptocurrency} Closing Prices Over Time ({interval})')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_df.index, y=data_df['Close'], mode='lines', name='Closing Price'))
    fig.update_layout(title=f'{cryptocurrency} Closing Prices Over Time ({interval})', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

    if data_df.empty:
        st.write(f"No data available for {cryptocurrency} with interval {interval}")
    else:
        scaled_data, scaler = preprocess_data(data_df)
        n_past = 60
        n_future = 1
        X, y = create_datasets(scaled_data, n_past, n_future)

        if len(X) == 0 or len(y) == 0:
            st.write(f"Not enough data to create datasets for {cryptocurrency} with interval {interval}")
        else:
            model = load_model(f'data/lstm_model_{cryptocurrency}_{interval}.keras')

            recent_data = scaled_data[-n_past:]
            recent_data = recent_data.reshape((1, n_past, 1))
            predicted_price = model.predict(recent_data)
            predicted_price = scaler.inverse_transform(predicted_price)

            st.subheader('Current Price and Predicted Price')
            st.write(f"**Current Price of {cryptocurrency}:** ${data_df['Close'].iloc[-1]:.2f}")
            st.write(f"**Predicted Price of {cryptocurrency}:** ${predicted_price[0, 0]:.2f}")
            
            st.subheader('Predicted vs Actual Prices')
            y_pred = model.predict(X)
            y_pred = scaler.inverse_transform(y_pred)
            y_test = scaler.inverse_transform(y)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.flatten(), mode='lines', name='Actual Price'))
            fig2.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred.flatten(), mode='lines', name='Predicted Price'))
            fig2.update_layout(title='Predicted vs Actual Prices', xaxis_title='Time', yaxis_title='Price')
            st.plotly_chart(fig2, use_container_width=True)


