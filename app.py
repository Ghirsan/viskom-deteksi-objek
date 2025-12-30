import streamlit as st
from ultralytics import YOLO
import PIL.Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load Model
model = YOLO('runs/detect/train2/weights/best.pt')

st.title("Sistem Deteksi Objek Real-time 🏍️")
st.sidebar.title("Pengaturan")

# Pilihan Input
source_radio = st.sidebar.radio("Pilih Sumber Input", ["Gambar", "Webcam (Real-time)"])

# AMBANG BATAS CONFIDENCE
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Image Input
if source_radio == "Gambar":
    uploaded_file = st.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        # Prediksi
        results = model.predict(image, conf=conf_threshold)
        # Hasil
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)

# Webcam Input
elif source_radio == "Webcam (Real-time)":
    st.write("Izinkan akses kamera pada browser untuk memulai deteksi.")

    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(img, conf=conf_threshold)
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Inisialisasi WebRTC Streamer dengan STUN Server yang lebih lengkap
    webrtc_streamer(
        key="yolo-detection",
        video_processor_factory=VideoProcessor,
        # Penambahan ICE Servers untuk menembus firewall jaringan
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )

    # Inisialisasi WebRTC Streamer
    webrtc_streamer(
        key="yolo-detection",
        video_transformer_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

st.sidebar.info("Model ini mendeteksi: Helmet, Person, Motorcycle")

