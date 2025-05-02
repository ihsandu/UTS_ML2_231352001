import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# Memuat dataset
df = pd.read_csv('Toyota_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Weekday'] = df['Date'].dt.day_name()

# Normalisasi data
scaler = MinMaxScaler()
numerical_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Menyiapkan fitur (X) dan target (y)
X = df[['Adj Close', 'High', 'Low', 'Open', 'Volume']]
y = df['Close']

# Membuat dan melatih model Linear Regression
model = LinearRegression()
model.fit(X, y)

# Menyimpan model ke file
joblib.dump(model, 'toyota_stock_model.pkl')

# Aplikasi Streamlit
st.title('Prediksi Harga Saham Toyota')

# Input data dari pengguna
adj_close = st.number_input('Harga Penutupan Terkoreksi', min_value=0.0)
high = st.number_input('Harga Tertinggi')
low = st.number_input('Harga Terendah')
open_price = st.number_input('Harga Pembukaan')
volume = st.number_input('Volume')

# Fungsi untuk melakukan simulasi prediksi dengan input pengguna
def model_simulation(model, adj_close, high, low, open_price, volume):
    input_data = np.array([[adj_close, high, low, open_price, volume]])
    prediksi_harga = model.predict(input_data)
    return prediksi_harga[0]

# Prediksi harga saham
predicted_price = model_simulation(model, adj_close, high, low, open_price, volume)
st.write(f"Prediksi Harga Penutupan: {predicted_price}")

# Menampilkan grafik harga saham Toyota
st.subheader('Grafik Harga Saham Toyota')

# Membuat figure dan axes untuk plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.plot(df['Date'], df['Close'], label='Harga Saham')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Saham')
ax.set_title('Pergerakan Harga Saham Toyota')
ax.grid(True)
ax.legend()

# Menampilkan figure ke Streamlit
st.pyplot(fig)