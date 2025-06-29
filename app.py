import streamlit as st
import joblib
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ========== SETUP ==========
st.set_page_config(page_title="Deteksi Berita Hoax", layout="wide", page_icon="📰")

# Load model dan vectorizer
model = joblib.load("logreg_hoax_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Logo
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("kemdikbud.png", width=100)
with col2:
    st.markdown("<h1 style='text-align: center; color: navy;'>Deteksi Berita Hoax</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: gray;'>Menggunakan Machine Learning Logistic Regression</h5>", unsafe_allow_html=True)
with col3:
    st.image("uhtp.png", width=100)

st.markdown("<hr style='border: 2px solid #0d6efd;'>", unsafe_allow_html=True)

# ========== FUNGSI CLEANING ==========
stopword_factory = StopWordRemoverFactory()
stopwords = stopword_factory.get_stop_words()
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"\\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    text = " ".join(tokens)
    text = stemmer.stem(text)
    return text

# ========== UI ==========
with st.container():
    st.markdown("### 📌 Masukkan teks berita atau pesan yang ingin diperiksa:")

    input_text = st.text_area("Contoh: 'Selamat Anda mendapatkan hadiah undian Rp100 juta dari Telkomsel.'", height=150)

    if st.button("🔍 Deteksi Hoax"):
        if input_text.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu.")
        else:
            cleaned = clean_text(input_text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            prob = model.predict_proba(vectorized)[0][prediction]

            if prediction == 1:
                st.error(f"❌ Prediksi: **HOAX**  \n🔴 Probabilitas: {prob:.2f}")
            else:
                st.success(f"✅ Prediksi: **VALID / BUKAN HOAX**  \n🟢 Probabilitas: {prob:.2f}")


# ========== FORM PELAPORAN HOAX ==========

st.markdown("---")
st.markdown("#### 📨 Laporkan Berita Hoax Baru")

st.markdown("Jika Anda menemukan berita hoax baru yang belum dikenali oleh sistem kami, mohon bantu kami untuk memperbarui database dengan mengisi form berikut:")

with st.form("form_laporan_hoax"):
    laporan_text = st.text_area("📝 Deskripsikan isi pesan atau berita hoax yang Anda temui:", height=150)
    pelapor_nama = st.text_input("👤 Nama Anda (opsional)")
    pelapor_email = st.text_input("📧 Alamat Email (opsional)")
    submitted = st.form_submit_button("Kirim Laporan")

    if submitted:
        if laporan_text.strip() == "":
            st.warning("Silakan isi deskripsi hoax sebelum mengirim laporan.")
        else:
            st.success("✅ Laporan Anda telah diterima. Terima kasih atas kontribusinya!")
            st.info("📌 *Mari sama-sama kita berantas informasi hoax*")

# ========== KONTEN TAMBAHAN ==========
st.markdown("---")
st.markdown("#### ℹ️ Tentang Aplikasi")
st.markdown("""
Aplikasi ini dibangun untuk mendeteksi kemungkinan berita hoax atau valid berdasarkan input teks menggunakan algoritma *Machine Learning Logistic Regression*. 
Model dilatih menggunakan data berita dan pesan yang telah dilabeli serta diproses menggunakan teknik pembersihan teks dan vektorisasi TF-IDF.

**Fitur:**
- Model ringan dan cepat
- Tidak membutuhkan GPU
- Cocok untuk edukasi dan verifikasi cepat berita

**Developer:** Universitas Hang Tuah Pekanbaru  
**Mitra:** Karang Taruna Kecamatan Kerinci Kanan
""")







st.markdown("<hr style='border: 1px solid lightgray;'>", unsafe_allow_html=True)
st.markdown("<center><small>© 2025 Deteksi Hoax ML - Tim Dosen Universitas Hang Tuah Pekanbaru</small></center>", unsafe_allow_html=True)
