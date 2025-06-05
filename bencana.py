import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data (assuming the file is uploaded or available locally)
# For Colab environment, you might need to adjust the path
# If running locally, make sure the CSV is in the same directory or specify the full path
try:
    df = pd.read_csv('Korban_bencana.csv', delimiter=';')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop() # Stop the app if data loading fails

# --- Sidebar for Navigation ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Dataset, Karakteristik, dan Visualisasi", "Pelatihan Model", "Prediksi"])

# --- Page 1: Dataset, Karakteristik, dan Visualisasi ---
if page == "Dataset, Karakteristik, dan Visualisasi":
    st.title("Dataset, Karakteristik, dan Visualisasi Data Korban Bencana")

    st.header("Dataset")
    st.write("Berikut adalah tampilan awal dari dataset:")
    st.dataframe(df.head())

    st.header("Karakteristik Data")
    st.write("Informasi ringkasan statistik:")
    st.write(df.describe())

    st.write("Informasi tipe data dan missing values:")
    st.write(df.info())

    st.header("Visualisasi Data")

    st.subheader("Distribusi Kolom Numerik")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribusi {col}')
            st.pyplot(plt)
    else:
        st.write("Tidak ada kolom numerik untuk visualisasi distribusi.")

    st.subheader("Count Plot Kolom Kategorial (Top 10)")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            plt.figure(figsize=(10, 5))
            sns.countplot(y=df[col].value_counts().nlargest(10).index, data=df[df[col].isin(df[col].value_counts().nlargest(10).index)])
            plt.title(f'Top 10 {col}')
            st.pyplot(plt)
    else:
        st.write("Tidak ada kolom kategorial untuk visualisasi count.")

    # Add more visualizations relevant to the data
    # Example: Relationship between two numerical columns
    # if 'Kolom_Numerik_1' in numeric_cols and 'Kolom_Numerik_2' in numeric_cols:
    #     st.subheader("Scatter Plot antara Kolom_Numerik_1 dan Kolom_Numerik_2")
    #     plt.figure(figsize=(8, 6))
    #     sns.scatterplot(x='Kolom_Numerik_1', y='Kolom_Numerik_2', data=df)
    #     st.pyplot(plt)


# --- Page 2: Pelatihan Model ---
elif page == "Pelatihan Model":
    st.title("Pelatihan Model")

    st.write("Bagian ini akan digunakan untuk melatih model machine learning.")
    st.write("Untuk contoh ini, kita akan membuat model regresi linier sederhana.")
    st.write("Pilih fitur (X) dan target (y) untuk model.")

    all_columns = df.columns.tolist()
    target_column = st.selectbox("Pilih Kolom Target (y)", all_columns)

    feature_columns = st.multiselect("Pilih Kolom Fitur (X)", [col for col in all_columns if col != target_column])

    if not feature_columns or not target_column:
        st.warning("Pilih setidaknya satu fitur dan satu target untuk melatih model.")
    else:
        # Ensure selected columns are numeric for Linear Regression
        X = df[feature_columns]
        y = df[target_column]

        numeric_X_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if not numeric_X_cols:
            st.error("Fitur yang dipilih tidak memiliki kolom numerik. Regresi Linier memerlukan input numerik.")
        elif y.dtype not in ['int64', 'float64']:
             st.error("Kolom target yang dipilih bukan numerik. Regresi Linier memerlukan target numerik.")
        else:
            X_numeric = X[numeric_X_cols] # Use only numeric features

            st.write(f"Menggunakan fitur numerik: {numeric_X_cols}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

            st.write("Data dibagi menjadi training (80%) dan testing (20%).")
            st.write(f"Ukuran data training: {X_train.shape[0]}")
            st.write(f"Ukuran data testing: {X_test.shape[0]}")

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            st.success("Model Linear Regression berhasil dilatih!")

            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("Evaluasi Model")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"R-squared (R2): {r2:.2f}")

            # You can display model coefficients etc.
            # st.subheader("Koefisien Model")
            # coefficients = pd.DataFrame(model.coef_, index=X_numeric.columns, columns=['Coefficient'])
            # st.dataframe(coefficients)

# --- Page 3: Prediksi ---
elif page == "Prediksi":
    st.title("Formulir Prediksi")

    st.write("Masukkan nilai untuk fitur-fitur di bawah ini untuk mendapatkan prediksi.")
    st.write("Catatan: Bagian ini memerlukan model yang sudah dilatih di halaman 'Pelatihan Model'.")

    # This part is more complex as it depends on the trained model from Page 2.
    # For a simple example, we'll simulate a prediction form.
    # In a real application, you would save and load the trained model.

    st.warning("Untuk demo ini, prediksi bersifat simulasi karena model tidak disimpan.")
    st.write("Jika model sudah dilatih, Anda bisa mengisi nilai fitur dan mendapatkan hasil prediksi.")

    # Example of input fields based on numeric features from Page 2
    # You would need to dynamically create input fields based on the features
    # selected for training in Page 2.
    st.subheader("Input Fitur")

    # Example input fields (replace with your actual feature names and types)
    # input_feature1 = st.number_input("Input untuk Fitur 1 (numerik)", value=0.0)
    # input_feature2 = st.number_input("Input untuk Fitur 2 (numerik)", value=0.0)
    # ... add more inputs for your features ...

    st.write("Formulir input prediksi akan ditampilkan di sini berdasarkan fitur yang digunakan untuk melatih model.")

    # Simulate prediction button
    # if st.button("Lakukan Prediksi"):
        # Create a DataFrame from the input values (matching the structure used for training)
        # input_data = pd.DataFrame([[input_feature1, input_feature2, ...]], columns=['Fitur1', 'Fitur2', ...])

        # Assume a trained model 'model' is available (in a real app, load it)
        # if 'model' in locals(): # Check if model was trained
        #     prediction = model.predict(input_data)
        #     st.subheader("Hasil Prediksi")
        #     st.write(f"Prediksi nilai target: {prediction[0]:.2f}")
        # else:
        #     st.error("Model belum dilatih. Silakan latih model di halaman 'Pelatihan Model' terlebih dahulu.")

# Note on running in Colab:
# To run this Streamlit app in Google Colab, you need to use `streamlit run app.py`.
# Colab does not directly display the Streamlit output in the notebook.
# You'll typically get a public URL to access the app.
# You might need to use libraries like `ngrok` or `localtunnel` to expose the port.
# Example using ngrok (run this in a new cell AFTER the %%writefile cell):
# !pip install pyngrok
# from pyngrok import ngrok
# public_url = ngrok.connect(8501)
# print(f"Streamlit app running at: {public_url}")
# !streamlit run app.py &>/dev/null&
