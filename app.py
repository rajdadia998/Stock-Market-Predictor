import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('Stock Predictions Model.keras')

# Define the header and input field with updated CSS
st.markdown("<h1 style='text-align: center; color: white; font-size: 36px;'>Stock Market Predictor</h1>", unsafe_allow_html=True)
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Fetch stock data
start = '2012-01-01'
end = '2022-12-31'
data = yf.download(stock, start, end)

# Display stock data
st.markdown("<h2 style='text-align: center; color: white; font-size: 24px;'>Stock Data</h2>", unsafe_allow_html=True)
st.write(data.style.set_table_styles([
    {'selector': 'table', 'props': [('border', '1px solid #ddd')]},
    {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-size', '16px'), ('padding', '12px 20px')]},
    {'selector': 'td', 'props': [('font-size', '14px'), ('padding', '8px 20px')]},
]).set_properties(**{'border-collapse': 'collapse', 'border': '1px solid #ddd'}))

# Prepare data for prediction
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Display Price vs MA50 plot
st.markdown("<h2 style='text-align: center; color: white; font-size: 24px;'>Price vs MA50</h2>", unsafe_allow_html=True)
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(14,8))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

# Display Price vs MA50 vs MA100 plot
st.markdown("<h2 style='text-align: center; color: white; font-size: 24px;'>Price vs MA50 vs MA100</h2>", unsafe_allow_html=True)
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(14,8))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Display Price vs MA100 vs MA200 plot
st.markdown("<h2 style='text-align: center; color: white; font-size: 24px;'>Price vs MA100 vs MA200</h2>", unsafe_allow_html=True)
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(14,8))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x,y = np.array(x), np.array(y)

# Make predictions
predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

# Display Original Price vs Predicted Price plot
st.markdown("<h2 style='text-align: center; color: white; font-size: 24px;'>Original Price vs Predicted Price</h2>", unsafe_allow_html=True)
fig4, ax4 = plt.subplots(figsize=(14,8))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
