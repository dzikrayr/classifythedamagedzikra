import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf  # Kembali ke TF normal
import numpy as np
from PIL import Image

# Konfigurasi Halaman (Harus dipanggil pertama kali)
st.set_page_config(
    page_title="ClassifyTheDamage: Deteksi Kerusakan Bangunan dan Jalan Akibat Bencana",
    page_icon="🏢",
    layout="centered"
)

components.html(
    """
    <script>
        const meta = window.parent.document.createElement('meta');
        meta.name = "dicoding:email";
        meta.content = "dzikra.yuhasyra9a@gmail.com";
        window.parent.document.getElementsByTagName('head')[0].appendChild(meta);
    </script>
    """,
    width=0, height=0
)

# Fungsi untuk memuat model (di-cache agar tidak dimuat ulang setiap kali ada interaksi)
@st.cache_resource
def load_model():
    # 1. Ganti tulisan di bawah dengan ID File Google Drive milikmu
    file_id = '1R7VJtTkWelp6_zINseBcPJ741-M4iEw6' 
    
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'Model_Utuh.keras' # Nama file saat disimpan di server
    
    # 2. Server akan mengecek: Kalau file belum ada, download dari Drive!
    if not os.path.exists(output):
        with st.spinner('Mengunduh model dari database untuk pertama kalinya... (Mohon tunggu)'):
            gdown.download(url, output, quiet=False)
    
    # 3. Load model yang sudah utuh
    model = tf.keras.models.load_model(output, compile=False)
    return model

model = load_model()

# Dictionary untuk memetakan output prediksi ke label yang mudah dibaca
# Sesuaikan dengan jumlah dan nama kelas pada datasetmu
CLASS_NAMES = {
    0: 'Damaged_building',
    1: 'Damaged_highway',
    2: 'Non-damaged_building',
    3: 'Non-damaged_highway',
    4: 'debris'
}

# --- SIDEBAR ---
st.sidebar.title("Tentang Aplikasi ℹ️")
st.sidebar.info(
    "Aplikasi ini menggunakan model Deep Learning "
    "untuk mengklasifikasikan tingkat kerusakan bangunan pascabencana "
    "dari sebuah foto."
)
st.sidebar.markdown("---")
st.sidebar.write("Dibuat menggunakan **Streamlit** dan **TensorFlow/Keras**.")

# --- HALAMAN UTAMA ---
st.title("🏢 Sistem Deteksi Kerusakan Bangunan dan Jalan")
st.write("Unggah foto bangunan dan jalan untuk menganalisis kondisinya.")

# Widget untuk mengunggah gambar
uploaded_file = st.file_uploader("Pilih file gambar (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    
    # Membuat dua kolom agar layout rapi
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    with col2:
        st.write("### Hasil Analisis:")
        
        # Tombol untuk mulai memprediksi
        if st.button("Analisis Gambar 🔍"):
            with st.spinner('Sedang memproses gambar...'):
                # Preprocessing Gambar
                # Sesuaikan target_size dengan input modelmu (misal: 224x224)
                img_resized = image.resize((150, 150)) 
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                # Normalisasi jika saat training kamu menormalisasi data (misal / 255.0)
                img_array = img_array / 255.0 

                # Prediksi
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100

                # Menampilkan Hasil
                label_hasil = CLASS_NAMES.get(predicted_class, "Tidak diketahui")
                
                st.success(f"**Klasifikasi:** {label_hasil}")
                st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")
                
                # Progress bar untuk visualisasi confidence
                st.progress(int(confidence))
