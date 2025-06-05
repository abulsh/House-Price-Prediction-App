import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os
import locale

# Set locale untuk format Rupiah
locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')

# Cek apakah model sudah ada, jika tidak buat dummy model
MODEL_PATH = "house_price_model.pkl"

if not os.path.exists(MODEL_PATH):
    # Buat dummy dataset dan model
    df = pd.DataFrame({
        'bedrooms': np.random.randint(1, 6, 100),
        'bathrooms': np.random.uniform(1, 4, 100),
        'sqft_living': np.random.randint(500, 4000, 100),
        'sqft_lot': np.random.randint(1000, 10000, 100),
        'floors': np.random.choice([1.0, 1.5, 2.0], 100),
        'waterfront': np.random.choice([0, 1], 100),
        'view': np.random.randint(0, 5, 100),
        'grade': np.random.randint(1, 14, 100),
        'yr_built': np.random.randint(1950, 2020, 100),
        'price': np.random.randint(100_000_000, 3_000_000_000, 100)
    })

    X = df.drop("price", axis=1)
    y = df["price"]

    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction App")

st.markdown("Prediksi harga rumah berdasarkan fitur-fitur properti.")

# Input fitur properti
st.header("Input Fitur Rumah")
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    sqft_living = st.number_input("Luas Bangunan (sqft)", min_value=300, max_value=10000, value=1500)
    floors = st.number_input("Jumlah Lantai", min_value=1.0, max_value=3.5, value=1.0, step=0.5)

with col2:
    sqft_lot = st.number_input("Luas Tanah (sqft)", min_value=500, max_value=20000, value=5000)
    waterfront = st.selectbox("Pemandangan Laut", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    view = st.slider("Pemandangan (0 = Buruk, 4 = Sangat Bagus)", min_value=0, max_value=4, value=0)
    grade = st.slider("Grade Bangunan (1-13)", min_value=1, max_value=13, value=7)

yr_built = st.number_input("Tahun Dibangun", min_value=1900, max_value=2025, value=2000)

# Buat DataFrame input
input_df = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'sqft_lot': [sqft_lot],
    'floors': [floors],
    'waterfront': [waterfront],
    'view': [view],
    'grade': [grade],
    'yr_built': [yr_built]
})

# Prediksi saat tombol diklik
if st.button("Prediksi Harga"):
    prediction = model.predict(input_df)[0]
    formatted_price = f"Rp {prediction:,.0f}".replace(",", ".")
    st.success(f"💰 Estimasi Harga Rumah: {formatted_price}")

st.markdown("---")
st.markdown("📌 *Prediksi ini berdasarkan model machine learning dan tidak menggantikan penilaian profesional.*")
