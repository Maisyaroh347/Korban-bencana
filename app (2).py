import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ==== Load Dataset ====
@st.cache_data
def load_data():
    return pd.read_csv("Korban_bencana.csv", sep=";", engine="python")

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Dataset tidak ditemukan. Pastikan file 'Korban_bencana.csv' tersedia.")
    st.stop()

# ==== Halaman 1: Dataset & Visualisasi ====
def page_dataset():
    st.title("Dataset dan Visualisasi Korban Bencana")

    st.subheader("Cuplikan Data")
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

    st.subheader("Tipe Data")
    st.write(df.dtypes)

    st.subheader("Distribusi Total Deaths")
    fig, ax = plt.subplots()
    sns.histplot(df["Total Deaths"].fillna(0), kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Korelasi Fitur Numerik")
    num_cols = df.select_dtypes(include="number").columns
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ==== Halaman 2: Pelatihan Model ====
def page_training():
    st.title("Pelatihan Model Prediksi Jumlah Korban Jiwa")

    if "Total Deaths" not in df.columns:
        st.error("Kolom 'Total Deaths' tidak ditemukan dalam dataset.")
        return

    df_clean = df.dropna(subset=["Total Deaths"])
    X = df_clean.select_dtypes(include="number").drop(columns=["Total Deaths"], errors="ignore")
    y = df_clean["Total Deaths"]

    if X.empty or y.empty:
        st.warning("Data kosong setelah dibersihkan.")
        return

    if st.button("Latih Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Model berhasil dilatih!")
        st.write(f"📉 Mean Absolute Error: {mae:.2f}")
        st.write(f"📈 R² Score: {r2:.2f}")

        # Simpan model
        joblib.dump(model, "model_rf.pkl")
        joblib.dump(X.columns.tolist(), "model_columns.pkl")
        st.success("Model dan fitur disimpan sebagai 'model_rf.pkl' dan 'model_columns.pkl'.")

# ==== Halaman 3: Formulir Prediksi ====
def page_prediction():
    st.title("Formulir Prediksi Jumlah Korban Jiwa")

    if not (os.path.exists("model_rf.pkl") and os.path.exists("model_columns.pkl")):
        st.warning("Model belum tersedia. Silakan latih model terlebih dahulu.")
        return

    model = joblib.load("model_rf.pkl")
    features = joblib.load("model_columns.pkl")

    st.write("Silakan isi data berikut:")

    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Prediksi"):
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi Total Deaths: {prediction:.2f}")

# ==== Navigasi ====
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", [
    "Dataset dan Visualisasi",
    "Pelatihan Model",
    "Formulir Prediksi"
])

if page == "Dataset dan Visualisasi":
    page_dataset()
elif page == "Pelatihan Model":
    page_training()
elif page == "Formulir Prediksi":
    page_prediction()
