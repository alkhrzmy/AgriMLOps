import os
import json
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
MODEL_METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", "models/model_v1_metadata.json"))

st.set_page_config(page_title="AgriMLOps PlantWild", page_icon="🌱", layout="wide")

st.title("AgriMLOps PlantWild")
st.caption("MVP diagnosis penyakit tanaman in-the-wild dengan feedback loop dan monitoring sederhana.")

page = st.sidebar.radio(
    "Menu",
    ["Diagnosis Tanaman", "Dashboard Monitoring", "Active Learning Queue", "Model Registry"],
)

if page == "Diagnosis Tanaman":
    st.header("Diagnosis Tanaman")
    uploaded_file = st.file_uploader("Upload foto tanaman", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Preview gambar", use_container_width=True)
        st.info("Endpoint prediksi akan diaktifkan pada tahap implementasi backend inference.")

elif page == "Dashboard Monitoring":
    st.header("Dashboard Monitoring")
    st.info("Monitoring prediksi, confidence, feedback, dan drift akan ditambahkan pada tahap berikutnya.")

elif page == "Active Learning Queue":
    st.header("Active Learning Queue")
    st.info("Antrian validasi data low-confidence dan feedback user akan ditambahkan pada tahap berikutnya.")

else:
    st.header("Model Registry")
    if MODEL_METADATA_PATH.exists():
        with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Version", metadata.get("model_version", "-"))
        col2.metric("Num Classes", metadata.get("num_classes", "-"))
        col3.metric("Input Size", metadata.get("input_size", "-"))

        st.subheader("Detail Model")
        st.write(
            {
                "model_name": metadata.get("model_name", "-"),
                "dataset": metadata.get("dataset", "-"),
                "accuracy": metadata.get("accuracy", "-"),
                "macro_f1": metadata.get("macro_f1", "-"),
                "created_at": metadata.get("created_at", "-"),
                "notes": metadata.get("notes", "-"),
            }
        )
    else:
        st.warning("Model metadata belum ditemukan. Jalankan training di Kaggle dan extract artifact ke folder models/.")

with st.sidebar:
    st.divider()
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.ok:
            st.success("API online")
        else:
            st.warning("API tidak merespons normal")
    except requests.RequestException:
        st.warning("API belum tersedia")
