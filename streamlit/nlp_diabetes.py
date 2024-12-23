import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import streamlit as st
from streamlit_option_menu import option_menu


# Load dataset chatbot
qa_data = pd.read_csv("diabetes_qa_dataset2.csv")
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(qa_data["Pertanyaan"])


# Fungsi chatbot
def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = similarity.argmax()
    if similarity[0, best_match] > 0.5:
        return qa_data["Jawaban"][best_match]
    else:
        return "Maaf, saya tidak mengerti pertanyaan Anda. Silakan coba lagi."


# Load model prediksi diabetes
model = joblib.load('diabetes_model.sav') 
scaler = joblib.load('scaler.pkl')

# Sidebar menu dengan ikon
with st.sidebar:
    selected = option_menu(
        "Navigasi Menu",
        ["Home", "Chatbot", "Prediksi Diabetes"],
        icons=["house", "chat-dots", "activity"],  # Ikon menggunakan Bootstrap Icons
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f0f0"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "padding": "10px"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )

# Tampilan berdasarkan menu
if selected == "Home":
    st.title("Selamat Datang di Aplikasi Diabetes Assistant")
    st.image("banner.jpg", use_container_width=True)
    st.write("""
        Aplikasi ini dirancang untuk membantu Anda memahami lebih banyak tentang diabetes 
        dan memprediksi risiko diabetes berdasarkan data kesehatan.
    """)

elif selected == "Chatbot":
    st.header("💬 Chatbot Tanya Jawab Seputar Diabetes")

    # Desain Chatbox
    st.markdown(
        """
        <style>
        .chatbox {
            max-width: 700px;
            padding: 10px;
            margin: 10px auto;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .user-bubble {
            background-color: #DCF8C6;
            padding: 10px;
            margin: 10px auto;
            border-radius: 10px;
            max-width: 700px;
        }
        .bot-bubble {
            background-color: #FFFFFF;
            padding: 10px;
            margin: 10px auto;
            border-radius: 10px;
            border: 1px solid #4CAF50;
            max-width: 700px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Input pengguna
    user_query = st.text_input("Ajukan pertanyaan Anda di sini:")

    if st.button("Kirim"):
        response = chatbot_response(user_query)

        # Tampilkan percakapan
        st.markdown(f"""
        <div class="chatbox">
            <div class="user-bubble"><strong>Anda:</strong> {user_query}</div>
            <div class="bot-bubble"><strong>Chatbot:</strong> {response}</div>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Prediksi Diabetes":
    st.title("📊 Prediksi Risiko Diabetes")
    st.write(
        """
        Masukkan informasi kesehatan Anda di bawah ini untuk memprediksi 
        risiko terkena diabetes.
        """
    )

    # Tambahkan styling untuk input dan hasil prediksi
    st.markdown(
        """
        <style>
        .input-section {
            border: 2px solid #4CAF50;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 20px;
        }
        .pred-button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            width: 100%;
            border: none;
        }
        .pred-button:hover {
            background-color: #45a049;
            cursor: pointer;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .high-risk {
            background-color: #ffcccc;
            border: 2px solid #ff0000;
            color: #ff0000;
        }
        .low-risk {
            background-color: #ccffcc;
            border: 2px solid #00b300;
            color: #00b300;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Input data kesehatan
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    # Input features sesuai urutan model
    pregnancies = st.number_input("🤰 Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=0)
    glucose = st.number_input("📈 Kadar Glukosa (Glucose)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("💉 Tekanan Darah (Blood Pressure)", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input("📏 Ketebalan Kulit (Skin Thickness)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("🧪 Kadar Insulin (Insulin)", min_value=0, max_value=900, value=85)
    bmi = st.number_input("⚖️ Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    dpf = st.number_input(
        "🔬 Diabetes Pedigree Function (DPF)", min_value=0.0, max_value=2.5, value=0.5, step=0.01
    )
    age = st.number_input("👤 Usia (Age)", min_value=0, max_value=120, value=30)

    st.markdown('</div>', unsafe_allow_html=True)

    # Validasi input
    if st.button("🔍 Prediksi Risiko Diabetes"):
        try:
            # Pastikan urutan fitur sesuai dengan saat training
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_data)  # Scaling data input
            
            prediction = model.predict(input_scaled)  # Prediksi kelas
            prediction_prob = model.predict_proba(input_scaled)  # Probabilitas kelas

            # Tampilkan hasil prediksi
            if prediction[0] == 1:
                st.markdown(
                    f"""
                    <div class="result-box high-risk">
                        <h4>🚨 Hasil Prediksi</h4>
                        <p><strong>Risiko Tinggi:</strong> Anda memiliki {prediction_prob[0][1]*100:.2f}% kemungkinan terkena diabetes.</p>
                        <p>Disarankan untuk berkonsultasi dengan dokter dan mulai menerapkan gaya hidup sehat. Berikut adalah beberapa tips untuk menjaga kesehatan Anda:
                        <p>1. Kurangi konsumsi gula dan karbohidrat sederhana.<br>
                        2. Tingkatkan aktivitas fisik, seperti berjalan kaki atau olahraga ringan.<br>
                        3. Perbanyak konsumsi sayuran, buah-buahan, dan makanan tinggi serat.<br>
                        4. Lakukan pemeriksaan kesehatan secara rutin untuk memantau kondisi tubuh Anda.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box low-risk">
                        <h4>✅ Hasil Prediksi</h4>
                        <p><strong>Risiko Rendah:</strong> Anda memiliki {prediction_prob[0][0]*100:.2f}% kemungkinan bebas dari diabetes.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {str(e)}")