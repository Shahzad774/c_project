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

# Streamlit app layout
st.set_page_config(page_title='Cryptocurrency Price Prediction', layout='wide')

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Prediction"],
        icons=["house", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
    )


# Home page
if selected == "Home":
    st.title('Cryptocurrency Price Prediction')
    st.image('crypto.jpg', use_column_width=True)  # Add the image here
    st.write("""
    This app predicts the future prices of cryptocurrencies using a Long Short-Term Memory (LSTM) model.
    Select a cryptocurrency from the sidebar to get started.
    """)

    # # Picture upload option
    # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption='Uploaded Image', use_column_width=True)

# Prediction page
if selected == "Prediction":
    st.title('Cryptocurrency Price Prediction')

    # Select cryptocurrency
    cryptocurrency = st.selectbox('Select Cryptocurrency', ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD'])
    interval = st.selectbox('Select Interval', ['1d', '1wk'])

    # Get today's date
    today = datetime.today().date()
    three_years_ago = today - timedelta(days=3*365)

    # Fetch data
    data_df = fetch_crypto_data(cryptocurrency, start=three_years_ago, end=today, interval=interval)
    
    # Plot data
    st.subheader(f'{cryptocurrency} Closing Prices Over Time ({interval})')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_df.index, y=data_df['Close'], mode='lines', name='Closing Price'))
    fig.update_layout(title=f'{cryptocurrency} Closing Prices Over Time ({interval})', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

    if data_df.empty:
        st.write(f"No data available for {cryptocurrency} with interval {interval}")
    else:
        # Preprocess data
        scaled_data, scaler = preprocess_data(data_df)
        n_past = 60
        n_future = 1
        X, y = create_datasets(scaled_data, n_past, n_future)

        if len(X) == 0 or len(y) == 0:
            st.write(f"Not enough data to create datasets for {cryptocurrency} with interval {interval}")
        else:
            # Load the model
            model = load_model(f'data/lstm_model_{cryptocurrency}_{interval}.keras')

            # Predict the latest price
            recent_data = scaled_data[-n_past:]
            recent_data = recent_data.reshape((1, n_past, 1))
            predicted_price = model.predict(recent_data)
            predicted_price = scaler.inverse_transform(predicted_price)

            # Show the current price and predicted price
            st.subheader('Current Price and Predicted Price')
            st.write(f"**Current Price of {cryptocurrency}:** ${data_df['Close'].iloc[-1]:.2f}")
            st.write(f"**Predicted Price of {cryptocurrency}:** ${predicted_price[0, 0]:.2f}")
            
            # Plot predictions
            st.subheader('Predicted vs Actual Prices')
            y_pred = model.predict(X)
            y_pred = scaler.inverse_transform(y_pred)
            y_test = scaler.inverse_transform(y)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.flatten(), mode='lines', name='Actual Price'))
            fig2.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred.flatten(), mode='lines', name='Predicted Price'))
            fig2.update_layout(title='Predicted vs Actual Prices', xaxis_title='Time', yaxis_title='Price')
            st.plotly_chart(fig2, use_container_width=True)



#  latest code

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from tensorflow.keras.models import load_model
# import yfinance as yf
# from datetime import datetime, timedelta
# from sklearn.preprocessing import MinMaxScaler
# from streamlit_option_menu import option_menu

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

# # Streamlit app layout
# st.set_page_config(page_title='Cryptocurrency Price Prediction', layout='wide')

# # Sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         menu_title="Main Menu",
#         options=["Home", "Prediction"],
#         icons=["house", "graph-up-arrow"],
#         menu_icon="cast",
#         default_index=0,
#     )

# # Home page
# if selected == "Home":
#     st.title('Cryptocurrency Price Prediction')
#     st.write("""
#     This app predicts the future prices of cryptocurrencies using a Long Short-Term Memory (LSTM) model.
#     Select a cryptocurrency from the sidebar to get started.
#     """)

# # Prediction page
# if selected == "Prediction":
#     st.title('Cryptocurrency Price Prediction')

#     # Select cryptocurrency
#     cryptocurrency = st.selectbox('Select Cryptocurrency', ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD'])

#     # Get today's date
#     today = datetime.today().date()
#     one_year_ago = today - timedelta(days=365)

#     # Fetch data
#     data_df = fetch_crypto_data(cryptocurrency, start=one_year_ago, end=today)
    
#     # Plot data
#     st.subheader(f'{cryptocurrency} Closing Prices Over Time')
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data_df.index, y=data_df['Close'], mode='lines', name='Closing Price'))
#     fig.update_layout(title=f'{cryptocurrency} Closing Prices Over Time', xaxis_title='Date', yaxis_title='Price')
#     st.plotly_chart(fig, use_container_width=True)

#     # Preprocess data
#     scaled_data, scaler = preprocess_data(data_df)
#     n_past = 60
#     n_future = 1
#     X, y = create_datasets(scaled_data, n_past, n_future)

#     # Load the model
#     model = load_model('data/lstm_model.keras')

#     # Predict the latest price
#     recent_data = scaled_data[-n_past:]
#     recent_data = recent_data.reshape((1, n_past, 1))
#     predicted_price = model.predict(recent_data)
#     predicted_price = scaler.inverse_transform(predicted_price)

#     # Show the current price and predicted price
#     st.subheader('Current Price and Predicted Price')
#     st.write(f"**Current Price of {cryptocurrency}:** ${data_df['Close'].iloc[-1]:.2f}")
#     st.write(f"**Predicted Price of {cryptocurrency}:** ${predicted_price[0, 0]:.2f}")
    
#     # Plot predictions
#     st.subheader('Predicted vs Actual Prices')
#     y_pred = model.predict(X)
#     y_pred = scaler.inverse_transform(y_pred)
#     y_test = scaler.inverse_transform(y)

#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.flatten(), mode='lines', name='Actual Price'))
#     fig2.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred.flatten(), mode='lines', name='Predicted Price'))
#     fig2.update_layout(title='Predicted vs Actual Prices', xaxis_title='Time', yaxis_title='Price')
#     st.plotly_chart(fig2, use_container_width=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# import yfinance as yf
# from streamlit_option_menu import option_menu

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
#     X = []
#     for i in range(n_past, len(data) - n_future + 1):
#         X.append(data[i - n_past:i, 0])
#     X = np.array(X)
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     return X

# # Load pre-trained model
# model = tf.keras.models.load_model('data/saved_model.keras')

# # Streamlit app layout
# st.set_page_config(page_title='Cryptocurrency Price Prediction', layout='wide')

# # Sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         menu_title="Main Menu",
#         options=["Home", "Prediction"],
#         icons=["house", "graph-up-arrow"],
#         menu_icon="cast",
#         default_index=0,
#     )

# # Home page
# if selected == "Home":
#     st.title('Cryptocurrency Price Prediction')
#     st.write("""
#     This app predicts the future prices of cryptocurrencies using a Long Short-Term Memory (LSTM) model.
#     Select a cryptocurrency from the sidebar to get started.
#     """)
#     st.image('images/banner.jpg', use_column_width=True)
#     st.write("""
#     #### How It Works:
#     1. **Fetch Data**: Get historical data for the selected cryptocurrency.
#     2. **Preprocess Data**: Scale the data to a range of 0 to 1 for better model performance.
#     3. **Load Pre-trained Model**: Use the LSTM model trained previously to make predictions.
#     4. **Predict**: Forecast future prices based on the learned patterns.
#     """)

# # Prediction page
# if selected == "Prediction":
#     st.title('Cryptocurrency Price Prediction')

#     # Select cryptocurrency
#     cryptocurrency = st.selectbox('Select Cryptocurrency', ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD'])

#     # Fetch data
#     data_df = fetch_crypto_data(cryptocurrency, '2011-01-01', '2024-05-23')

#     # Plot data
#     st.subheader(f'{cryptocurrency} Closing Prices Over Time')
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data_df.index, y=data_df['Close'], mode='lines', name='Closing Price'))
#     fig.update_layout(title=f'{cryptocurrency} Closing Prices Over Time', xaxis_title='Date', yaxis_title='Price')
#     st.plotly_chart(fig, use_container_width=True)

#     # Preprocess data
#     scaled_data, scaler = preprocess_data(data_df)
#     n_past = 60
#     n_future = 1
#     X = create_datasets(scaled_data, n_past, n_future)

#     # Predict
#     y_pred_scaled = model.predict(X)
#     y_pred = scaler.inverse_transform(y_pred_scaled)

#     # Display prediction
#     actual_price = data_df['Close'].values[-1]
#     predicted_price = y_pred[-1][0]
#     st.subheader(f'Predicted Price for {cryptocurrency}')
#     st.write(f'The predicted price for the next day is: ${predicted_price:.2f}')

#     # Additional paragraphs for predicted vs actual prices
#     st.markdown(f"""
#     <div style="background-color:#f0f0f5; padding:10px; border-radius:5px;">
#         <h3 style="color:#333;">Actual and Predicted Prices</h3>
#         <p style="color:#333;">The actual price for {cryptocurrency} as of the last available data point is <strong>${actual_price:.2f}</strong>.</p>
#         <p style="color:#333;">The model predicts that the price for the next day will be <strong>${predicted_price:.2f}</strong>.</p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown(f"""
#     <div style="background-color:#e0f7fa; padding:10px; border-radius:5px; margin-top:20px;">
#         <h3 style="color:#00796b;">Predicted Values Summary</h3>
#         <p style="color:#00796b;">Based on historical data and the LSTM model, the predicted values give an insight into the potential future trends of the cryptocurrency market. 
#         The prediction made here is for one day ahead, but the model can be extended to predict further into the future with additional training and data adjustments.</p>
#     </div>
#     """, unsafe_allow_html=True)

#     # Additional plot for predicted vs actual
#     st.subheader('Predicted vs Actual Prices')
#     fig2 = go.Figure()
#     actual_prices = data_df['Close'].values[-len(y_pred):]
#     fig2.add_trace(go.Scatter(x=list(range(len(actual_prices))), y=actual_prices, mode='lines', name='Actual Price'))
#     fig2.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred.flatten(), mode='lines', name='Predicted Price'))
#     fig2.update_layout(title='Predicted vs Actual Prices', xaxis_title='Time', yaxis_title='Price')
#     st.plotly_chart(fig2, use_container_width=True)




# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# import yfinance as yf
# from streamlit_option_menu import option_menu

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
#     X = []
#     for i in range(n_past, len(data) - n_future + 1):
#         X.append(data[i - n_past:i, 0])
#     X = np.array(X)
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     return X

# # Load pre-trained model
# model = tf.keras.models.load_model('data/saved_model.keras')

# # Streamlit app layout
# st.set_page_config(page_title='Cryptocurrency Price Prediction', layout='wide')

# # Sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         menu_title="Main Menu",
#         options=["Home", "Prediction"],
#         icons=["house", "graph-up-arrow"],
#         menu_icon="cast",
#         default_index=0,
#     )

# # Home page
# if selected == "Home":
#     st.title('Cryptocurrency Price Prediction')
#     st.write("""
#     This app predicts the future prices of cryptocurrencies using a Long Short-Term Memory (LSTM) model.
#     Select a cryptocurrency from the sidebar to get started.
#     """)
#     st.image('images/banner.jpg', use_column_width=True)
#     st.write("""
#     #### How It Works:
#     1. **Fetch Data**: Get historical data for the selected cryptocurrency.
#     2. **Preprocess Data**: Scale the data to a range of 0 to 1 for better model performance.
#     3. **Load Pre-trained Model**: Use the LSTM model trained previously to make predictions.
#     4. **Predict**: Forecast future prices based on the learned patterns.
#     """)

# # Prediction page
# if selected == "Prediction":
#     st.title('Cryptocurrency Price Prediction')

#     # Select cryptocurrency
#     cryptocurrency = st.selectbox('Select Cryptocurrency', ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD'])

#     # Fetch data
#     data_df = fetch_crypto_data(cryptocurrency, '2011-01-01', '2024-05-23')

#     # Plot data
#     st.subheader(f'{cryptocurrency} Closing Prices Over Time')
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data_df.index, y=data_df['Close'], mode='lines', name='Closing Price'))
#     fig.update_layout(title=f'{cryptocurrency} Closing Prices Over Time', xaxis_title='Date', yaxis_title='Price')
#     st.plotly_chart(fig, use_container_width=True)

#     # Preprocess data
#     scaled_data, scaler = preprocess_data(data_df)
#     n_past = 60
#     n_future = 1
#     X = create_datasets(scaled_data, n_past, n_future)

#     # Predict
#     y_pred = model.predict(X[-1].reshape(1, n_past, 1))
#     y_pred = scaler.inverse_transform(y_pred)

#     # Display prediction
#     st.subheader(f'Predicted Price for {cryptocurrency}')
#     st.write(f'The predicted price for the next day is: ${y_pred[0][0]:.2f}')

#     # Additional plot for predicted vs actual (if required)
#     st.subheader('Predicted vs Actual Prices')
#     fig2 = go.Figure()
#     actual_prices = data_df['Close'].values[-len(y_pred):]
#     predicted_prices = scaler.inverse_transform(model.predict(X[-len(y_pred):]))
#     fig2.add_trace(go.Scatter(x=list(range(len(actual_prices))), y=actual_prices, mode='lines', name='Actual Price'))
#     fig2.add_trace(go.Scatter(x=list(range(len(predicted_prices))), y=predicted_prices.flatten(), mode='lines', name='Predicted Price'))
#     fig2.update_layout(title='Predicted vs Actual Prices', xaxis_title='Time', yaxis_title='Price')
#     st.plotly_chart(fig2, use_container_width=True)












# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.model_selection import train_test_split
# import yfinance as yf
# from streamlit_option_menu import option_menu

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

# # Streamlit app layout
# st.set_page_config(page_title='Cryptocurrency Price Prediction', layout='wide')

# # Sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         menu_title="Main Menu",
#         options=["Home", "Prediction"],
#         icons=["house", "graph-up-arrow"],
#         menu_icon="cast",
#         default_index=0,
#     )

# # Home page
# if selected == "Home":
#     st.title('Cryptocurrency Price Prediction')
#     st.write("""
#     This app predicts the future prices of cryptocurrencies using a Long Short-Term Memory (LSTM) model.
#     Select a cryptocurrency from the sidebar to get started.
#     """)

# # Prediction page
# if selected == "Prediction":
#     st.title('Cryptocurrency Price Prediction')

#     # Select cryptocurrency
#     cryptocurrency = st.selectbox('Select Cryptocurrency', ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD'])

#     # Fetch data
#     data_df = fetch_crypto_data(cryptocurrency, '2011-01-01', '2024-05-23')

#     # Plot data
#     st.subheader(f'{cryptocurrency} Closing Prices Over Time')
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data_df.index, y=data_df['Close'], mode='lines', name='Closing Price'))
#     fig.update_layout(title=f'{cryptocurrency} Closing Prices Over Time', xaxis_title='Date', yaxis_title='Price')
#     st.plotly_chart(fig, use_container_width=True)

#     # Preprocess data
#     scaled_data, scaler = preprocess_data(data_df)
#     n_past = 60
#     n_future = 1
#     X, y = create_datasets(scaled_data, n_past, n_future)

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Build and train the model
#     model = build_model((n_past, 1), n_future)
#     history = model.fit(X_train, y_train, batch_size=1, epochs=2, validation_split=0.2)

#     # Predict
#     y_pred = model.predict(X_test)
#     y_pred = scaler.inverse_transform(y_pred)
#     y_test = scaler.inverse_transform(y_test)

#     # Plot predictions
#     st.subheader('Predicted vs Actual Prices')
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.flatten(), mode='lines', name='Actual Price'))
#     fig2.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred.flatten(), mode='lines', name='Predicted Price'))
#     fig2.update_layout(title='Predicted vs Actual Prices', xaxis_title='Time', yaxis_title='Price')
#     st.plotly_chart(fig2, use_container_width=True)
