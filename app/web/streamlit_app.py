import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("API_URL", "http://localhost:8000")
CURRENT_MODEL_VERSION = os.getenv("CURRENT_MODEL_VERSION", "v1")
MODEL_METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", "models/model_v1_metadata.json"))

st.set_page_config(page_title="AgriMLOps PlantWild", page_icon="🌱", layout="wide")

st.title("AgriMLOps PlantWild")
st.caption("MVP diagnosis penyakit tanaman in-the-wild dengan feedback loop dan monitoring sederhana.")

page = st.sidebar.radio(
    "Menu",
    ["Diagnosis Tanaman", "Dashboard Monitoring", "Active Learning Queue", "Model Registry"],
)


def api_get(path: str) -> dict | None:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Gagal menghubungi API: {exc}")
        return None


def load_model_registry_rows(current_version: str) -> list[dict]:
    rows = []
    for version in ["v1", "v2"]:
        metadata_path = Path(f"models/model_{version}_metadata.json")
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        rows.append(
            {
                "version": version,
                "status": "deployed" if version == current_version else "candidate",
                "model_name": metadata.get("model_name", "-"),
                "accuracy": metadata.get("accuracy", "-"),
                "macro_f1": metadata.get("macro_f1", "-"),
                "feedback_samples_used": metadata.get("feedback_samples_used", "-"),
                "created_at": metadata.get("created_at", "-"),
            }
        )
    return rows


if page == "Diagnosis Tanaman":
    st.header("Diagnosis Tanaman")
    uploaded_file = st.file_uploader("Upload foto tanaman", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Preview gambar", use_container_width=True)
        if st.button("Prediksi Penyakit", type="primary"):
            with st.spinner("Menjalankan inference model..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=120)
                    response.raise_for_status()
                    st.session_state["last_prediction"] = response.json()
                except requests.RequestException as exc:
                    st.error(f"Prediksi gagal: {exc}")

    prediction = st.session_state.get("last_prediction")
    if prediction:
        st.subheader("Hasil Prediksi")
        col1, col2, col3 = st.columns(3)
        col1.metric("Label", prediction.get("predicted_label", "-"))
        col2.metric("Confidence", f"{prediction.get('confidence', 0.0):.2%}")
        col3.metric("Model", prediction.get("model_version", "-"))

        if prediction.get("needs_review"):
            st.warning("Confidence rendah. Prediksi ini masuk active learning queue untuk validasi.")

        st.write("**Top-3 Prediksi**")
        top3 = prediction.get("top3_predictions", [])
        if top3:
            st.dataframe(pd.DataFrame(top3), use_container_width=True)

        st.write("**Rekomendasi Tindakan**")
        st.info(prediction.get("recommendation", "-"))

        st.subheader("Feedback User")
        with st.form("feedback_form"):
            feedback_label = st.radio(
                "Apakah prediksi ini benar?",
                ["Prediksi benar", "Prediksi salah", "Tidak yakin"],
                horizontal=True,
            )
            suggested_label = st.text_input("Suggested label/koreksi label")
            comment = st.text_area("Catatan")
            submitted = st.form_submit_button("Kirim Feedback")

        if submitted:
            feedback_map = {
                "Prediksi benar": "correct",
                "Prediksi salah": "incorrect",
                "Tidak yakin": "unsure",
            }
            payload = {
                "prediction_id": prediction["prediction_id"],
                "user_feedback": feedback_map[feedback_label],
                "suggested_label": suggested_label or None,
                "comment": comment or None,
            }
            try:
                response = requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=20)
                response.raise_for_status()
                st.success("Feedback berhasil disimpan.")
            except requests.RequestException as exc:
                st.error(f"Gagal mengirim feedback: {exc}")

elif page == "Dashboard Monitoring":
    st.header("Dashboard Monitoring")
    summary = api_get("/monitoring/summary")
    if summary:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Predictions", summary.get("total_predictions", 0))
        col2.metric("Avg Confidence", f"{summary.get('average_confidence', 0.0):.2%}")
        col3.metric("Low Confidence", summary.get("low_confidence_count", 0))
        col4.metric("Feedback", summary.get("feedback_count", 0))
        col5.metric("Pending AL", summary.get("active_learning_pending_count", 0))

        predictions_by_label = summary.get("predictions_by_label", {})
        st.subheader("Distribusi Prediksi per Label")
        if predictions_by_label:
            label_df = pd.DataFrame(
                [{"label": label, "count": count} for label, count in predictions_by_label.items()]
            ).set_index("label")
            st.bar_chart(label_df)
        else:
            st.info("Belum ada prediksi.")

        st.subheader("Histogram Confidence")
        histogram = summary.get("confidence_histogram", {})
        if histogram:
            hist_df = pd.DataFrame([{"bucket": key, "count": value} for key, value in histogram.items()]).set_index("bucket")
            st.bar_chart(hist_df)

elif page == "Active Learning Queue":
    st.header("Active Learning Queue")
    queue_response = api_get("/active-learning/queue")
    items = queue_response.get("items", []) if queue_response else []
    if not items:
        st.info("Tidak ada item pending.")
    else:
        st.dataframe(pd.DataFrame(items), use_container_width=True)
        st.subheader("Validasi Item")
        for item in items:
            with st.expander(f"{item['predicted_label']} | {item['reason']} | {item['confidence']:.2%}"):
                st.write(f"Prediction ID: `{item['prediction_id']}`")
                st.write(f"Queue ID: `{item['id']}`")
                if item.get("image_path"):
                    image_path = Path(item["image_path"])
                    if image_path.exists():
                        st.image(str(image_path), caption="Uploaded image", use_container_width=True)
                with st.form(f"validate_{item['id']}"):
                    validated_label = st.text_input("Validated label", value=item.get("validated_label") or "")
                    validator_note = st.text_area("Validator note", value=item.get("validator_note") or "")
                    status = st.selectbox("Status", ["validated", "rejected"])
                    submitted = st.form_submit_button("Simpan Validasi")
                if submitted:
                    payload = {
                        "validated_label": validated_label or None,
                        "validator_note": validator_note or None,
                        "status": status,
                    }
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/active-learning/{item['id']}/validate",
                            json=payload,
                            timeout=20,
                        )
                        response.raise_for_status()
                        st.success("Item berhasil divalidasi.")
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(f"Gagal validasi item: {exc}")

else:
    st.header("Model Registry")
    metadata = api_get("/model/current")
    if not metadata and MODEL_METADATA_PATH.exists():
        with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    if metadata:
        current_version = metadata.get("current_model_version") or metadata.get("model_version") or CURRENT_MODEL_VERSION
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Model Version", current_version)
        col2.metric("Num Classes", metadata.get("num_classes", "-"))
        col3.metric("Input Size", metadata.get("input_size", "-"))

        registry_rows = load_model_registry_rows(current_version)
        if registry_rows:
            st.subheader("Registered Artifacts")
            st.dataframe(pd.DataFrame(registry_rows), use_container_width=True)

        st.subheader("Detail Model")
        st.write(
            {
                "model_name": metadata.get("model_name", "-"),
                "model_version": metadata.get("model_version", "-"),
                "dataset": metadata.get("dataset", "-"),
                "accuracy": metadata.get("accuracy", "-"),
                "macro_f1": metadata.get("macro_f1", "-"),
                "feedback_samples_used": metadata.get("feedback_samples_used", "-"),
                "created_at": metadata.get("created_at", "-"),
                "notes": metadata.get("notes", "-"),
            }
        )
    else:
        st.warning("Model metadata belum ditemukan. Jalankan training di Kaggle dan extract artifact ke folder models/.")

with st.sidebar:
    st.divider()
    st.caption(f"API: {API_BASE_URL}")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.ok:
            health = response.json()
            st.success(f"API online | Model: {health.get('model_version')}")
        else:
            st.warning("API tidak merespons normal")
    except requests.RequestException:
        st.warning("API belum tersedia")
