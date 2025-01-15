import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Judul aplikasi
st.title("Aplikasi Prediksi Penyakit Jantung untuk Praktisi Kesehatan")

st.write("Aplikasi ini dirancang untuk membantu praktisi kesehatan dalam memprediksi kemungkinan seorang pasien terindikasi penyakit jantung berdasarkan data medis yang tersedia.")

st.header("Input Data Pasien")

# Input data pasien
age = st.number_input("Umur", min_value=1, max_value=120, value=30)
sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
chest_pain = st.selectbox("Jenis Nyeri Dada", ["Angina Stabil", "Angina Tidak Stabil", "Asimptomatik", "Nyeri Dada Lainnya"])
resting_bp = st.number_input("Tekanan Darah Saat Istirahat (mmHg)", min_value=50, max_value=200, value=120)
cholesterol = st.number_input("Kolesterol (mg/dL)", min_value=100, max_value=400, value=200)
fasting_bs = st.selectbox("Gula Darah Puasa (> 120 mg/dL)", ["Ya", "Tidak"])
resting_ecg = st.selectbox("Hasil EKG Saat Istirahat", ["Normal", "Kelainan Gelombang ST", "Hipertrofi Ventrikel Kiri"])
max_hr = st.number_input("Detak Jantung Maksimum", min_value=60, max_value=220, value=100)
exercise_angina = st.selectbox("Angina Selama Latihan", ["Ya", "Tidak"])
oldpeak = st.number_input("Depresi ST (Oldpeak)", min_value=0.0, max_value=10.0, value=0.0)
st_slope = st.selectbox("Kemiringan Segmen ST", ["Meningkat", "Datar", "Menurun"])

# Mapping inputs to model format
# Pastikan LabelEncoder sudah dibuat untuk fitur-fitur kategorikal
label_encoders = {
    "ChestPainType": LabelEncoder().fit(["Angina Stabil", "Angina Tidak Stabil", "Asimptomatik", "Nyeri Dada Lainnya"]),
    "RestingECG": LabelEncoder().fit(["Normal", "Kelainan Gelombang ST", "Hipertrofi Ventrikel Kiri"]),
    "ST_Slope": LabelEncoder().fit(["Meningkat", "Datar", "Menurun"])
}

input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [1 if sex == "Laki-laki" else 0],
    "ChestPainType": [label_encoders["ChestPainType"].transform([chest_pain])[0]],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [1 if fasting_bs == "Ya" else 0],
    "RestingECG": [label_encoders["RestingECG"].transform([resting_ecg])[0]],
    "MaxHR": [max_hr],
    "ExerciseAngina": [1 if exercise_angina == "Ya" else 0],
    "Oldpeak": [oldpeak],
    "ST_Slope": [label_encoders["ST_Slope"].transform([st_slope])[0]]
})

# Load model
try:
    model = joblib.load("model_heart_disease.pkl")
except FileNotFoundError:
    st.error("Model tidak ditemukan. Harap unggah model_heart_disease.pkl terlebih dahulu.")
    model = None

# Prediction
if st.button("Prediksi"):
    if model:
        prediction = model.predict(input_data)[0]
        st.header("Hasil Prediksi")
        if prediction == 1:
            st.error("Pasien mungkin memiliki risiko penyakit jantung.")
        else:
            st.success("Pasien kemungkinan tidak memiliki risiko penyakit jantung yang signifikan.")
    else:
        st.error("Model belum siap digunakan.")