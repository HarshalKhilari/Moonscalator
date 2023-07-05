import os, sys, math, random, time, datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import yfinance as yf

from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter


import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential

import pmdarima as pm
from dateutil.relativedelta import relativedelta
from pmdarima.metrics import smape



# CachedLimiterSession class extends the CacheMixin, LimiterMixin, and Session classes.
# This class combines request caching (SQLiteCache) and rate limiting (Limiter) functionalities to prevent excessive API requests.
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

# An instance of CachedLimiterSession is created as session with a rate limit of 2 requests per 5 seconds and a caching backend using SQLite.
session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache")
    )


# suppresses the output to the standard output stream (stdout). 
# This class is used to silence the printing of unwanted information during the execution of the script.
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



# Function to calculate directional accuracy from actual and predicted stock values
# to understand how well our models capture trends
def calculate_directional_accuracy(actual_prices, predicted_prices):
    # Ensure the lengths of actual_prices and predicted_prices are the same
    if len(actual_prices) != len(predicted_prices):
        raise ValueError("Lengths of actual_prices and predicted_prices must be the same.")
    correct_predictions = 0
    total_predictions = len(actual_prices) - 1  # Exclude the first price for direction comparison
    for i in range(1, len(actual_prices)):
        actual_direction = actual_prices[i] - actual_prices[i-1]
        predicted_direction = predicted_prices[i] - actual_prices[i-1]
        if (actual_direction >= 0 and predicted_direction >= 0) or (actual_direction < 0 and predicted_direction < 0):
            correct_predictions += 1
    directional_accuracy = correct_predictions / total_predictions
    return directional_accuracy
        



# AUTO - ARIMA MODEL
    
def auto_arima_model(y_train, y_test, forecast_days):
# automatically computes all parameters and gives predictions
    # Creating a dataframe to store forecasts
    fc_df = pd.DataFrame(columns = ['fc'], index = forecast_days)
    auto = pm.auto_arima(
                     y = y_train, 
                     max_order = None,
                     seasonal=False, 
                     stepwise=True,
                     maxiter = 100,
                     suppress_warnings=True, 
                     error_action="ignore",
                     trace=True
                        )
    model = auto  # seeded from the model we've already fit
    def predict_next():
        fc = model.predict(n_periods=1, return_conf_int=False)
        return fc.tolist()[0]
    
    forecasts = []
    
    for date, price in y_test.items():
        fc = predict_next()
        forecasts.append(fc)
        # Updates the existing model with a small number of MLE steps
        model.update(pd.Series([price], index = [date], name = 'Close'))
    
    errors = {
        "MSE": mean_squared_error(y_test, forecasts),
        "SMAPE": smape(y_test, forecasts),
        "DA": calculate_directional_accuracy(y_test, forecasts)
    }
    
    for new_day in fc_df.index.tolist():
        fc = predict_next()
        fc_df.loc[new_day, 'fc'] = fc
        model.update(pd.Series([fc], index = [new_day], name = 'Close'))
    
    return fc_df, errors


# LSTM MODEL

def lstm_model(y_train, y_test, forecast_days):
# Long Short-Term Memory neural network to calculate forecast
    
    trainset = y_train.values
    testset = y_test.values

    # Scale our data from 0 to 1
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(np.concatenate((trainset, testset), axis=0).reshape(-1, 1))
    y_train_scaled = scaler.transform(trainset.reshape(-1, 1))
    y_test_scaled = scaler.transform(testset.reshape(-1, 1))
    
    # Use our scaled data for training
    train_X = []
    train_y = []

    for i in range(60, len(y_train_scaled)):
        train_X.append(y_train_scaled[i-60:i, 0])
        train_y.append(y_train_scaled[i, 0])

    train_X, train_y = np.array(train_X), np.array(train_y)

    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape = (train_X.shape[1], 1)))
    model.add(Dropout(0.35))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Тrain the model
    model.fit(train_X, train_y, batch_size=60, epochs=10)
    
    y_test_scaled = np.concatenate([y_train_scaled[-60:], y_test_scaled], axis = 0)
    
    # Create test dataset
    test_X = []
    for i in range(60, len(y_test_scaled)):
        test_X.append(y_test_scaled[i-60:i, 0])

    test_X = np.array(test_X)

    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1 ))

    # Predict on test data
    predictions = model.predict(test_X)
    predictions = scaler.inverse_transform(predictions)

    errors = {
        "MSE": mean_squared_error(y_test, predictions),
        "SMAPE": smape(y_test, predictions),
        "DA": calculate_directional_accuracy(y_test, predictions)
    }
    
    # Predict the stock prices for the forecast interval
    last_sequence = test_X[-1]  # Use the last sequence from the training data
    forecast = []

    for _ in range(len(forecast_days)):
        next_pred = model.predict(last_sequence.reshape(1, 60, -1))
        forecast.append(next_pred[0])
        last_sequence = np.append(last_sequence[1:], next_pred[0])

    # Inverse transform the forecasted data to obtain actual stock prices
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Creating a dataframe to store forecasts
    fc_df = pd.DataFrame(columns = ['fc'], index = forecast_days)

    i=0
    for new_day in fc_df.index.tolist():
        fc_df.loc[new_day, 'fc'] = forecast[i][0]
        i+=1
    
    return fc_df, errors




# METROPOLIS HASTINGS ALGORITHM (MARKOV CHAIN MONTE CARLO) 

mu, sig, N = 0.25, 0.5, 10000

def q(x):
    return (1 / (math.sqrt(2 * math.pi * sig ** 2))) * (math.e ** (-((x - mu) ** 2) / (2 * sig ** 2)))

def MCMC(n):   
    r = np.zeros(1)
    p = q(r[0])
    pts = []

    for i in range(N):
        rn = r + np.random.uniform(-1, 1)
        pn = q(rn[0])
        if pn >= p:
            p = pn
            r = rn
        else:
            u = np.random.rand()
            if u < pn / p:
                p = pn
                r = rn
        pts.append(r)

    pts = random.sample(pts, len(pts))
    pts = np.array(pts)
    
    return pts

def MH(y_train, y_test, is_forecast = False):
    y_test = np.array(y_test)
    stock_pred = []
    maturnity = 1
    volatility = 0.25
    risk_free = 0.1
    timestep = 1
    steps = len(y_test)
    delta_t = maturnity / steps
    i = 0
    stock_pred.append(y_train[-1])
    while timestep < steps:
        stock_price = stock_pred[-i]
        time_exp = maturnity - delta_t * timestep
        # Generate z_t using MCMC method
        ptss = MCMC(N)
        stock_price = stock_price * math.exp(((risk_free - 0.5 * (
            math.pow(volatility, 2))) * delta_t + volatility * math.sqrt(delta_t) * ptss[timestep + 5]))
        stock_pred.append(stock_price)
        i = i + 1
        timestep = timestep + 1
    print(y_test.shape, np.array(stock_pred).shape)
    if not is_forecast:
        errors = {
        "MSE": mean_squared_error(y_test, stock_pred),
        "SMAPE": smape(y_test, stock_pred),
        "DA": calculate_directional_accuracy(y_test, stock_pred)
        }
    else:
        errors = np.nan
    
    return errors, stock_pred
    
def MCMC_model(y_train, y_test, forecast_days):

    # Creating a dataframe to store forecasts
    fc_df = pd.DataFrame(columns = ['fc'], index = forecast_days)
    
    val_errors, val_pred = MH(y_train, y_test)
    hist_data = np.concatenate((y_train, y_test), axis=0).flatten()
    
    errors, forecast = MH(hist_data, fc_df, is_forecast = True)
    
    i=0
    for new_day in fc_df.index.tolist():
        fc_df.loc[new_day, 'fc'] = forecast[i]
        i+=1
    
    return fc_df, val_errors



# EXPLORING DATA THROUGH PLOTS

# Candlestick plot
def plot_candlestick(data):
    # Create a candlestick trace
    trace = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )
    # Create a layout object
    layout = go.Layout(
        legend_orientation="h",
        legend=dict(x=.5, xanchor="center"),
        plot_bgcolor='#FFFFFF',  
        title = "Candlestick Plot for the last year",
        xaxis=dict(gridcolor = 'lightgrey', 
                   type='date', 
                   tickmode='auto', 
                   tickformat='%d-%m-%Y', 
                   showgrid=True),
        yaxis=dict(gridcolor = 'lightgrey'),
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=True
    )       
    # Create a figure object that contains the trace and layout
    fig = go.Figure(data=[trace], layout=layout)
    # Display the figure
    st.plotly_chart(fig)    

    
# Plot Relative Strength Index
def plot_rsi(data, window=14):
    # Input: stock data, window
    # output: RSI
    # Function to calculate RSI using historical closing prices of a stock
    close_delta = data['Close'].diff()
    up = close_delta.where(close_delta > 0, 0)
    down = -close_delta.where(close_delta < 0, 0)
    # Calculate the average gains and losses
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    # Calculate the relative strength (RS)
    rs = avg_gain / avg_loss
    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))
    # Create traces for the dotted lines at RSI 30 and 70
    trace_rsi_30 = go.Scatter(
        x=data.index,
        y=[30] * data.shape[0],
        mode='lines',
        line=dict(color='green', dash='dash'),
        name='OVERSOLD'
    )
    trace_rsi_70 = go.Scatter(
        x=data.index,
        y=[70] * data.shape[0],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='OVERBOUGHT'
    )
    # Create a scatter trace for the RSI values
    trace = go.Scatter(
        x=data.index,
        y=rsi,
        mode='lines',
        name='RSI'
    )
    # Create a layout object
    layout = go.Layout(
        plot_bgcolor='#FFFFFF',  
        title = "Relative Strength Index (RSI) for the last year",
        xaxis=dict(gridcolor = 'lightgrey', 
                   type='date', 
                   tickmode='auto', 
                   tickformat='%d-%m-%Y', 
                   showgrid=True),
        yaxis=dict(gridcolor = 'lightgrey'),
        xaxis_title="Date",
        yaxis_title="RSI",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=True
    )        
    # Create a figure object that contains the trace and layout
    fig = go.Figure(data=[trace, trace_rsi_30, trace_rsi_70], layout=layout)
    # Display the figure
    st.plotly_chart(fig)

# Define a function to calculate MACD
def plot_macd(data, short_period=12, long_period=26, signal_period=9):
    # Calculate the short-term EMA
    ema_short = data['Close'].ewm(span=short_period, adjust=False).mean()
    # Calculate the long-term EMA
    ema_long = data['Close'].ewm(span=long_period, adjust=False).mean()
    # Calculate the MACD line
    macd_line = ema_short - ema_long
    # Calculate the signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    # Calculate the MACD histogram
    macd_histogram = macd_line - signal_line
    # Create trace for the MACD line
    trace_macd = go.Scatter(
        x=data.index,
        y=macd_line,
        mode='lines',
        name='MACD'
    )
    # Create trace for the MACD signal line
    trace_signal = go.Scatter(
        x=data.index,
        y=signal_line,
        mode='lines',
        name='Signal Line'
    )
    # Create trace for the MACD histogram
    trace_histogram = go.Bar(
        x=data.index,
        y=macd_histogram,
        name='Histogram',
        marker=dict(color=[('green' if val >= 0 else 'red') for val in macd_histogram])
    )
    # Create a layout object
    layout = go.Layout(
        plot_bgcolor='#FFFFFF',  
        title = "Moving Average Convergence Divergence (MACD) for the last year",
        xaxis=dict(gridcolor = 'lightgrey', 
                   type='date', 
                   tickmode='auto', 
                   tickformat='%d-%m-%Y', 
                   showgrid=True),
        yaxis=dict(gridcolor = 'lightgrey'),
        xaxis_title="Date",
        yaxis_title="MACD",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=True
    )    
    # Create a figure object that contains the traces and layout
    fig = go.Figure(data=[trace_macd, trace_signal, trace_histogram], layout=layout)
    # Display the figure
    st.plotly_chart(fig)

# Define a function to see relation between volume and price
def plot_org_data(data):
    # Initialize the MinMaxScaler
    vol_scaler = MinMaxScaler(feature_range=(0.1, 2.5))
    # Reshape the volume data to a 2D array
    volume_data = data['Volume'].values.reshape(-1, 1)
    # Scale the volume data
    scaled_volume = vol_scaler.fit_transform(volume_data)
    # Update the 'Volume' column in the DataFrame with the scaled values
    scaled_vol = scaled_volume.flatten()
    # Create a scatter trace with volume as the marker size
    trace = go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines+markers',
        marker=dict(
            size=scaled_vol,
            sizemode='diameter',
            sizeref=0.1,
            sizemin=1,
            color=data['Volume'],
            colorscale='redor',
            showscale=True
        ),
        line=dict(color='#000000'),
        text=data['Volume'],
        hovertemplate='Volume: %{text}<br>Price: %{y:.2f}<extra></extra>',
    )
    # Create a layout object
    layout = go.Layout(
        legend_orientation="h",
        legend=dict(x=.5, xanchor="center"),
        plot_bgcolor='#FFFFFF',  
        title = "Volume for stock price for past year",
        xaxis=dict(gridcolor = 'lightgrey', 
                   type='date', 
                   tickmode='auto', 
                   tickformat='%d-%m-%Y', 
                   showgrid=True),
        yaxis=dict(gridcolor = 'lightgrey'),
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=True
    )
    # Create a figure object that contains the trace and layout
    fig = go.Figure(data=[trace], layout=layout)
    # Display the figure
    st.plotly_chart(fig)

# Plot entire historical data
def plot_hist_data(data):
    # Create a scatter trace
    trace = go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines'
    )
    # Create a layout object
    layout = go.Layout(
        legend_orientation="h",
        legend=dict(x=.5, xanchor="center"),
        plot_bgcolor='#FFFFFF',
        title='Entire Historical Data',
        xaxis=dict(gridcolor = 'lightgrey', 
                                 type='date', 
                                 tickmode='auto', 
                                 tickformat='%d-%m-%Y', 
                                 showgrid=True),
        yaxis=dict(gridcolor = 'lightgrey'),
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=True
    )
    # Create a figure object that contains the trace and layout
    fig = go.Figure(data=[trace], layout=layout)
    # Display the figure
    st.plotly_chart(fig)



# GETTING THE FINAL FORECAST

def get_forecast(hist, validation_days = 90, days_to_forecast = 30):

    # Getting the latest date from the dataframe
    last_day = hist.index[-1]
    first_day = last_day - relativedelta(years = 2)

    first_eda_day = last_day - relativedelta(years = 1)

    eda_df = hist.loc[first_eda_day:last_day, :]

    st.header("Exploring the data...")

    # Plot entire historical data
    plot_hist_data(hist)
    
    # Plot Price by Volume
    plot_org_data(eda_df)
    
    # Plot candelstick plot for the last year
    plot_candlestick(eda_df)

    # Calculate and plot RSI using the close price
    plot_rsi(eda_df)

    # Calculate and plot MACD using the close price and a window of 14 periods
    plot_macd(eda_df)

    # Getting the last training day based on the passed valudation days
    last_train_day = last_day - relativedelta(days = validation_days)

    # Getting training and testing test
    y_train = hist.loc[first_day:last_train_day, 'Close']
    y_test = hist.loc[last_train_day:, 'Close']

    all_days = hist.loc[first_day:, 'Close']

    # Getting the last forecast day
    last_forecast_day = last_day + relativedelta(days = days_to_forecast)

    # Getting data range for forecast days
    forecast_days = pd.date_range(start = last_day + relativedelta(days = 1), end = last_forecast_day, freq="B").tolist()

    start_time = time.time()
    fc_arima, errors_arima = auto_arima_model(y_train, y_test, forecast_days)
    print(f"ARIMA took {time.time() - start_time} seconds")
    start_time = time.time()
    fc_lstm, errors_lstm = lstm_model(y_train, y_test, forecast_days)
    print(f"LSTM took {time.time() - start_time} seconds")
    start_time = time.time()
    fc_mcmc, errors_mcmc = MCMC_model(y_train, y_test, forecast_days)
    print(f"MCMC took {time.time() - start_time} seconds")

    error_df = pd.DataFrame([errors_arima, errors_lstm, errors_mcmc], index = ['ARIMA', 'LSTM', 'MCMC'])
    
    print("Errors",error_df)

    forecasted_price=(fc_arima['fc'] + fc_lstm['fc'] + fc_mcmc['fc'])/3
    st.write("---")
    st.header("Forecasting Data for the next month...")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_days.index, y=all_days.values, mode='lines+markers', name='Historical Prices'))
    fig.add_trace(go.Scatter(x=fc_arima.index, y=forecasted_price, mode='lines+markers', 
                             line=dict(color='#FFB300'), marker = dict(size = 8), name='Forecast'))
    
    fig.update_layout(
                      legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      plot_bgcolor='#FFFFFF',  
                      xaxis=dict(gridcolor = 'lightgrey', 
                                 type='date', 
                                 tickmode='auto', 
                                 tickformat='%d-%m-%Y', 
                                 showgrid=True),
                      yaxis=dict(gridcolor = 'lightgrey'),
                      xaxis_title="Time",
                      yaxis_title="Stock price",
                      margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_rangeslider_visible=True)
    # fig.show()
    st.plotly_chart(fig)

    forecasted_price = pd.DataFrame({'Date':fc_arima.index.strftime('%d-%m-%Y'), 
                                     'Forecasted Price':forecasted_price.astype(int)})
    forecasted_price.set_index('Date', inplace = True)
    forecasted_price.rename_axis(None, inplace = True)

    st.write("---")
    st.write("Actual Forecast Prices...")
    # Display the centered DataFrame
    display_centered_dataframe(forecasted_price)
    


# STREAMLIT CODE

import streamlit as st

def display_centered_dataframe(dataframe):
    # Apply CSS styling to center-align the table
    centered_html = f'<div style="display: flex; justify-content: center;">{dataframe.to_html()}</div>'
    # Display the centered DataFrame using Streamlit
    st.markdown(centered_html, unsafe_allow_html=True)

def display_ticker_news(ticker):
    # Fetch news articles using yfinance
    news = ticker.news
    # Display news articles in Streamlit
    st.subheader(f"Latest News Articles for {ticker.info['longName']}")
    for article in news:
        st.markdown(f"**[{article['title']}]({article['link']})**")
        st.write(f"At {datetime.datetime.fromtimestamp(article['providerPublishTime'])}")
        st.write(f"Source: {article['publisher']}")
        st.write('   ')


st.set_page_config(page_title="Forecasting", page_icon="📈")

st.sidebar.header("Forecast for the next month")

st.title('Stock Price Forecast')
symbol = st.text_input(label = "Enter the stock ticker", placeholder = "Enter the stock ticker here...", key="symbol", label_visibility = 'collapsed')
if symbol:
    with st.spinner("Fetching stock data..."):
        # Create a yfinance ticker object for user-defined symbol
        stock = yf.Ticker(symbol)
        # Get entire available stock data using yfinance for user-defined symbol
        hist = stock.history(period="max")
    if hist.shape[0] == 0:
        st.error("No data found. Symbol is not listed.")
    else:
        with st.spinner("Calculating forecast..."):
            get_forecast(hist)
            st.write("---")
            display_ticker_news(stock)