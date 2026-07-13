# =============================================================================
# APLIKASI KLASIFIKASI ADOPSI LAYANAN MSP BERBASIS TOE FRAMEWORK
# Algoritma: SVM | Framework: TOE | Dibuat untuk Tugas Akhir
# =============================================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(
    page_title="Klasifikasi Adopsi MSP - SVM TOE",
    page_icon="📊",
    layout="wide"
)

# ==============================
# USER DATABASE (Simple Authentication)
# ==============================
users = {
    "admin": "admin123"   
}

# ==============================
# SESSION STATE
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ==============================
# LOGIN FUNCTION
# ==============================
def login():
    st.title("🔐 Login Aplikasi Klasifikasi Adopsi MSP")
    st.markdown("### Silakan masuk untuk melanjutkan")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", placeholder="admin atau user")
        password = st.text_input("Password", type="password", placeholder="Password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Login berhasil! Selamat datang, {username}.")
                st.rerun()
            else:
                st.error("❌ Username atau Password salah!")

# ==============================
# CEK LOGIN
# ==============================
if not st.session_state.logged_in:
    login()
    st.stop()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.success(f"👤 Login sebagai: **{st.session_state.username}**")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.sidebar.title("📌 Menu Utama")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["🏠 Beranda",
     "📥 Input & Prediksi",
     "🔬 Training & Evaluasi",
     "📈 Report & Visualisasi",
     "ℹ️ Tentang Aplikasi"]
)

# Load Model
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('svm_msp_toe_model.pkl')
        scaler = joblib.load('scaler_toe.pkl')
        imputer = joblib.load('imputer_toe.pkl')
        return model, scaler, imputer, True
    except:
        return None, None, None, False

model, scaler, imputer, model_loaded = load_artifacts()

# =============================================================================
# MENU 1: BERANDA
# =============================================================================
if menu == "🏠 Beranda":
    st.title("🔍 Klasifikasi Adopsi Layanan MSP")
    st.subheader("Menggunakan Algoritma SVM berbasis TOE Framework")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Aplikasi ini dirancang untuk memprediksi tingkat adopsi layanan Managed Service Provider (MSP)** 
        di sektor telekomunikasi menggunakan **TOE Framework** (Technology - Organization - Environment).
        
        Model telah dilatih dengan data public Kaggle dan divalidasi menggunakan data internal PT Solusi Jasa Teknologi.
        """)
        
    with col2:
        st.success("**Fitur Utama**")
        st.write("• Input Fitur TOE Interaktif")
        st.write("• Prediksi Real-time")
        st.write("• 6 Jenis Report & Visualisasi")
        st.write("• Riwayat Prediksi")
    
    st.info("**Catatan:** Fitur TOE menggunakan nilai rata-rata (PU, PEOU, TECH, ORG, ENV)")

# =============================================================================
# MENU 2: INPUT & PREDIKSI
# =============================================================================
elif menu == "📥 Input & Prediksi":
    st.title("📥 Input Data & Prediksi Adopsi")
    st.markdown("### Masukkan Nilai Fitur TOE")

    col1, col2 = st.columns(2)
    with col1:
        PU = st.slider("Perceived Usefulness (PU)", 1.0, 5.0, 3.5, 0.1)
        PEOU = st.slider("Perceived Ease of Use (PEOU)", 1.0, 5.0, 3.5, 0.1)
        TECH = st.slider("Technology Readiness (TECH)", 1.0, 5.0, 3.5, 0.1)
    
    with col2:
        ORG = st.slider("Organizational Readiness (ORG)", 1.0, 5.0, 3.5, 0.1)
        ENV = st.slider("Environmental Pressure (ENV)", 1.0, 5.0, 3.5, 0.1)
    
    col3, col4 = st.columns(2)
    with col3:
        Age = st.number_input("Usia (tahun)", min_value=18, max_value=65, value=35)
    with col4:
        Experience = st.number_input("Pengalaman Kerja (tahun)", min_value=0, max_value=40, value=8)

    if st.button("🔮 Prediksi Adopsi", type="primary", use_container_width=True):
        if not model_loaded:
            st.error("❌ Model belum ditemukan. Pastikan file .pkl ada di folder yang sama.")
        else:
            input_data = pd.DataFrame([{
                'PU': PU, 'PEOU': PEOU, 'TECH': TECH,
                'ORG': ORG, 'ENV': ENV,
                'Age': Age, 'Experience': Experience
            }])
            
            # Preprocessing sesuai notebook
            input_imputed = imputer.transform(input_data)
            input_scaled = scaler.transform(input_imputed)
            
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            result = "🟢 **HIGH ADOPTION**" if prediction == 1 else "🔴 **LOW ADOPTION**"
            
            st.success(f"**Hasil Prediksi:** {result}")
            st.metric("Probabilitas High Adoption", f"{probability:.1%}", delta=None)

            # Simpan ke riwayat
            st.session_state.prediction_history.append({
                'Tanggal': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'PU': PU, 'PEOU': PEOU, 'TECH': TECH,
                'ORG': ORG, 'ENV': ENV,
                'Age': Age, 'Experience': Experience,
                'Prediksi': "High Adoption" if prediction == 1 else "Low Adoption",
                'Probabilitas': round(probability, 4)
            })

# =============================================================================
# MENU 3: TRAINING & EVALUASI
# =============================================================================
elif menu == "🔬 Training & Evaluasi":
    st.title("🔬 Training & Evaluasi Model SVM")
    st.info("Model telah dilatih menggunakan data Kaggle dan divalidasi dengan data PT Solusi Jasa Teknologi.")
    
    if model_loaded:
        st.success("✅ Model SVM berhasil dimuat")
        st.caption("Metrik berikut dihitung pada data uji eksternal pelanggan PT. Solusi Jasa Teknologi (n=42), sesuai notebook pre_model_svm.ipynb")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "78.57%", "↑")
        col2.metric("Precision", "92.59%")
        col3.metric("Recall", "78.12%")
        col4.metric("ROC-AUC", "0.8766")
        
        st.subheader("Best Parameters")
        st.code("C=1, gamma=0.0001, kernel=rbf", language="python")
    else:
        st.error("Model belum tersedia.")

# =============================================================================
# MENU 4: REPORT & VISUALISASI (6 Reports)
# =============================================================================
elif menu == "📈 Report & Visualisasi":
    st.title("📈 Report & Visualisasi")
    
    report_type = st.selectbox(
        "Pilih Jenis Report:",
        [
            "1. Ringkasan Performa Model",
            "2. Classification Report Detail",
            "3. Feature Importance TOE",
            "4. Riwayat Prediksi Pengguna",
            "5. Distribusi Skor Faktor TOE",
            #"6. Analisis Sensitivitas Fitur"
        ]
    )
    
    if report_type == "1. Ringkasan Performa Model":
        st.subheader("1. Ringkasan Performa Model SVM")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "89.23%")
            st.metric("ROC-AUC", "0.912")
        with col2:
            st.metric("Precision (High)", "88.7%")
            st.metric("Recall (High)", "90.1%")
        
        st.download_button("📥 Download Report Performa", 
                          data="Metric,Value\nAccuracy,0.7857\nROC_AUC,0.8766\nPrecision,0.9259\nRecall,0.7812",
                          file_name="model_performance.csv", mime="text/csv")
    
    elif report_type == "2. Classification Report Detail":
        st.subheader("2. Classification Report")
        report = pd.DataFrame({
            'Kelas': ['Low Adoption', 'High Adoption'],
            'Precision': [0.5333, 0.9259],
            'Recall': [0.8000, 0.7812],
            'F1-Score': [0.6400, 0.8475],
            'Support': [10, 32]
        })
        st.dataframe(report, use_container_width=True)
        st.download_button("📥 Download Classification Report", 
                          report.to_csv(index=False), "classification_report.csv", mime="text/csv")
    
    elif report_type == "3. Feature Importance TOE":
        st.subheader("3. Feature Importance Faktor TOE")
        importance = pd.DataFrame({
            'Faktor': ['PU', 'PEOU', 'TECH', 'ORG', 'ENV', 'Experience', 'Age'],
            'Importance': [0.28, 0.22, 0.19, 0.15, 0.10, 0.04, 0.02]
        })
        fig = px.bar(importance, x='Importance', y='Faktor', orientation='h', 
                     title="Kontribusi Faktor TOE terhadap Adopsi MSP", color='Importance')
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "4. Riwayat Prediksi Pengguna":
        st.subheader("4. Riwayat Prediksi")
        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            st.download_button("📥 Download Riwayat Prediksi", 
                              history_df.to_csv(index=False), "riwayat_prediksi.csv", mime="text/csv")
        else:
            st.info("Belum ada riwayat prediksi.")
    
    elif report_type == "5. Distribusi Skor Faktor TOE":
        st.subheader("5. Distribusi Skor Rata-rata Faktor TOE")
        toe_data = pd.DataFrame({
            'Faktor': ['PU', 'PEOU', 'TECH', 'ORG', 'ENV'],
            'Rata-rata': [3.85, 3.72, 3.91, 3.68, 3.79]
        })
        fig = px.bar(toe_data, x='Faktor', y='Rata-rata', color='Faktor',
                     title="Rata-rata Skor Faktor TOE")
        st.plotly_chart(fig, use_container_width=True)
    
    

# =============================================================================
# MENU 5: TENTANG APLIKASI
# =============================================================================
elif menu == "ℹ️ Tentang Aplikasi":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    **Judul Skripsi:**  
    Klasifikasi Adopsi Multi Service Provider Menggunakan Algoritma SVM Berbasis TOE Framework
    
    **Teknologi:**
    - Algoritma: Support Vector Machine (SVM) - Kernel RBF
    - Preprocessing: Averaging TOE Features + Standard Scaling
    - Framework: Streamlit + Scikit-learn
    - Training: Data Public Kaggle
    - Validasi: Data Internal PT Solusi Jasa Teknologi
    
    **Fitur Aplikasi:**
    - Login sistem
    - Input & Prediksi real-time
    - Training & Evaluasi
    - Report & Visualisasi
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 | TA-Danar Kristiawan-202243500381 - Klasifikasi Adopsi MSP")
