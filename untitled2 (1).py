import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# ========== Baca Dataset ==========
@st.cache_data
def load_data():
    df = pd.read_csv("Korban_bencana.csv", sep=";", engine="python")
    return df

df = load_data()

# ========== Navigasi ==========
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Dataset & Visualisasi", "Pelatihan Model", "Formulir Prediksi"]
)

# ========== Halaman 1 ==========
if page == "Dataset & Visualisasi":
    st.title("Halaman 1: Dataset dan Visualisasi")
    
    st.subheader("Tampilan Data")
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

    st.subheader("Tipe Data")
    st.write(df.dtypes)

    st.subheader("Visualisasi Korelasi Numerik")
    num_cols = df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Distribusi Total Deaths")
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Total Deaths"].fillna(0), bins=30, kde=True)
    st.pyplot(plt.gcf())
    plt.clf()

# ========== Halaman 2 ==========
elif page == "Pelatihan Model":
    st.title("Halaman 2: Pelatihan Model")
    st.markdown("Model: **Random Forest Regressor** untuk memprediksi jumlah `Total Deaths`.")

    if "Total Deaths" not in df.columns:
        st.error("Kolom target 'Total Deaths' tidak ditemukan.")
        st.stop()

    # Preprocessing
    df_clean = df.dropna(subset=["Total Deaths"])
    X = df_clean.select_dtypes(include=np.number).drop("Total Deaths", axis=1)
    y = df_clean["Total Deaths"]

    if X.empty or y.empty:
        st.warning("Data kosong setelah dibersihkan.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("✅ Model berhasil dilatih!")

    st.subheader("Evaluasi Model")
    st.write(f"📉 Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")

    # Simpan model dan kolom
    st.session_state["model"] = model
    st.session_state["features"] = X.columns.tolist()

# ========== Halaman 3 ==========
elif page == "Formulir Prediksi":
    st.title("Halaman 3: Formulir Prediksi")
    
    if "model" not in st.session_state or "features" not in st.session_state:
        st.warning("⚠️ Model belum dilatih. Silakan ke halaman 'Pelatihan Model' terlebih dahulu.")
        st.stop()

    model = st.session_state["model"]
    features = st.session_state["features"]

    st.write("Isi data berikut untuk memprediksi jumlah korban meninggal (Total Deaths).")

    input_data = {}
    for feature in features:
        value = st.number_input(f"{feature}", value=0.0)
        input_data[feature] = value

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"Prediksi Jumlah Korban Jiwa: {prediction:.2f}")
