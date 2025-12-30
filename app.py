import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np

# Load model kustom Anda
model = YOLO('runs/detect/train2/weights/best.pt')

st.title("Sistem Deteksi Objek Real-time 🏍️")
st.sidebar.title("Pengaturan")

# Pilihan Input
source_radio = st.sidebar.radio("Pilih Sumber Input", ["Gambar", "Webcam (Real-time)"])

# AMBANG BATAS CONFIDENCE
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

if source_radio == "Gambar":
    uploaded_file = st.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        # Prediksi
        results = model.predict(image, conf=conf_threshold)
        # Tampilkan Hasil
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Hasil Deteksi", use_column_width=True)

elif source_radio == "Webcam (Real-time)":
    st.write("Klik tombol 'Start' di bawah untuk mulai deteksi.")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    
    # Inisialisasi webcam
    camera = cv2.VideoCapture(0)

    while run:
        success, frame = camera.read()
        if not success:
            st.error("Gagal mengakses webcam.")
            break
        
        # Konversi warna ke RGB (Streamlit butuh RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Jalankan Prediksi
        results = model.predict(frame_rgb, conf=conf_threshold)
        
        # Gambar bounding box
        annotated_frame = results[0].plot()
        
        # Update frame di web
        FRAME_WINDOW.image(annotated_frame)
    else:
        st.write("Webcam Berhenti.")
        camera.release()