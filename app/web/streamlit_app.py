import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from team_logo_base64 import TEAM_LOGO_BASE64, TEAM_LOGO_MIME
API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("API_URL", "http://localhost:8000")
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/e/ef/Logo_ITERA.png"
APP_TITLE = "AgriMLOps PlantWild"
APP_TAGLINE = "Aplikasi web untuk diagnosis penyakit tanaman dari foto, dilengkapi feedback, monitoring, dan model registry."
TEAM_NAME = "Janji Mau Fokus TA Biar Cepet Lulus"
st.set_page_config(page_title=APP_TITLE, page_icon="🌱", layout="wide")

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:wght@600;700&display=swap');

:root {
    --ink: #101915;
    --muted: #4f5d57;
    --accent: #2f7d5c;
    --accent-2: #8cc84b;
    --paper: #f7f6f2;
    --card: #ffffff;
    --shadow: 0 18px 45px rgba(12, 26, 20, 0.12);
    --ring: rgba(47, 125, 92, 0.18);
}

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--ink);
}

.stApp {
    background: radial-gradient(circle at 12% 18%, #e7f1e7 0%, #f7f6f2 45%, #eef2ec 100%);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(47, 125, 92, 0.08) 1px, transparent 1px);
    background-size: 18px 18px;
    opacity: 0.55;
    pointer-events: none;
    z-index: 0;
}

section.main > div {
    position: relative;
    z-index: 1;
}

h1, h2, h3, h4 {
    font-family: 'Fraunces', serif !important;
    letter-spacing: 0.2px;
    color: #0e1712;
}

.app-header {
    background: var(--card);
    border-radius: 24px;
    padding: 28px 30px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(16, 25, 21, 0.06);
}

.app-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.app-subtitle {
    font-size: 1rem;
    color: var(--muted);
    margin-bottom: 14px;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 6px 14px;
    background: rgba(47, 125, 92, 0.12);
    color: #1b4e3a;
    font-weight: 600;
    font-size: 0.85rem;
}

.hero {
    margin: 22px 0 8px;
    padding: 18px 22px;
    background: linear-gradient(130deg, rgba(47, 125, 92, 0.12), rgba(140, 200, 75, 0.18));
    border-radius: 20px;
    border: 1px solid rgba(47, 125, 92, 0.12);
    animation: fadeUp 0.7s ease;
}

.hero-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.hero-body {
    color: var(--muted);
    line-height: 1.55;
}

.info-card {
    background: var(--card);
    border-radius: 18px;
    padding: 16px 18px;
    border: 1px solid rgba(16, 25, 21, 0.06);
    box-shadow: 0 12px 30px rgba(14, 23, 18, 0.08);
    min-height: 112px;
}

.info-card .card-title {
    font-weight: 600;
    margin-bottom: 6px;
}

.info-card .card-body {
    color: var(--muted);
    font-size: 0.92rem;
}

.section-lead {
    color: var(--muted);
    margin: 0 0 16px;
}

.logo-box {
    text-align: right;
}

.logo-caption {
    color: var(--muted);
    font-size: 0.75rem;
    margin-top: 6px;
}

.section-divider {
    height: 1px;
    margin: 18px 0 28px;
    background: linear-gradient(90deg, rgba(47, 125, 92, 0.2), rgba(16, 25, 21, 0));
}

div[data-testid="metric-container"] {
    background: var(--card);
    border-radius: 16px;
    padding: 12px 16px;
    border: 1px solid rgba(16, 25, 21, 0.06);
    box-shadow: 0 10px 24px rgba(14, 23, 18, 0.08);
}

button[kind="primary"], .stButton > button {
    border-radius: 999px !important;
    border: 1px solid rgba(47, 125, 92, 0.2) !important;
    box-shadow: 0 10px 24px rgba(47, 125, 92, 0.22) !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(47, 125, 92, 0.08), rgba(140, 200, 75, 0.08));
    border-right: 1px solid rgba(16, 25, 21, 0.05);
}

.footer-note {
    text-align: center;
    color: var(--muted);
    margin-top: 40px;
    font-size: 0.82rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


def api_get(path: str, show_error: bool = True) -> dict | None:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        if show_error:
            st.error(f"Gagal menghubungi API: {exc}")
        return None


def render_team_logo_inline(width_px: int = 72) -> None:
    if not TEAM_LOGO_BASE64.strip():
        return
    logo_b64 = "".join(TEAM_LOGO_BASE64.split())
    st.markdown(
        f"""
        <img
            src="data:{TEAM_LOGO_MIME};base64,{logo_b64}"
            style="width: {width_px}px; max-width: 100%; border-radius: 12px;"
        />
        """,
        unsafe_allow_html=True,
    )


def render_team_logo() -> None:
    if not TEAM_LOGO_BASE64.strip():
        st.info("Logo tim belum tersedia.")
        return
    logo_b64 = "".join(TEAM_LOGO_BASE64.split())
    st.markdown(
        f"""
        <img
            src="data:{TEAM_LOGO_MIME};base64,{logo_b64}"
            style="width: 100%; max-width: 360px; border-radius: 16px; box-shadow: 0 12px 30px rgba(14, 23, 18, 0.08);"
        />
        """,
        unsafe_allow_html=True,
    )


header_col, team_logo_col, logo_col = st.columns([0.68, 0.16, 0.16], vertical_alignment="center")
with header_col:
        st.markdown(
                f"""
                <div class="app-header">
                        <div class="app-title">{APP_TITLE}</div>
                        <div class="app-subtitle">{APP_TAGLINE}</div>
                        <div class="badge-row">
                                <span class="badge">AI Diagnosis</span>
                                <span class="badge">Active Learning</span>
                                <span class="badge">Monitoring & Feedback</span>
                        </div>
                </div>
                """,
                unsafe_allow_html=True,
        )
        with team_logo_col:
            st.markdown("<div class=\"logo-box\">", unsafe_allow_html=True)
            render_team_logo_inline(width_px=74)
            st.markdown("</div>", unsafe_allow_html=True)
with logo_col:
        st.markdown("<div class=\"logo-box\">", unsafe_allow_html=True)
        st.image(LOGO_URL, width=90)
        st.markdown("<div class=\"logo-caption\">Institut Teknologi Sumatera</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style=\"height:16px\"></div>", unsafe_allow_html=True)

        summary = api_get("/monitoring/summary", show_error=False)
        if summary:
            total_predictions = summary.get("total_predictions", 0)
            avg_confidence = summary.get("average_confidence", 0.0)
            pending_al = summary.get("active_learning_pending_count", 0)

            cards = st.columns(3)
            with cards[0]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-title">Total Predictions</div>
                        <div class="card-body"><strong>{total_predictions}</strong> prediksi tercatat.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cards[1]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-title">Avg Confidence</div>
                        <div class="card-body"><strong>{avg_confidence:.2%}</strong> rata-rata confidence.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cards[2]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-title">Pending Active Learning</div>
                        <div class="card-body"><strong>{pending_al}</strong> item menunggu validasi.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown("<div class=\"section-divider\"></div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Menu",
    [
        "Diagnosis Tanaman",
        "Dashboard Monitoring",
        "Active Learning Queue",
        "Model Registry",
        "Tentang",
    ],
)

with st.sidebar:
    st.caption("Gunakan menu di atas untuk berpindah fitur utama.")


if page == "Diagnosis Tanaman":
    st.header("Diagnosis Tanaman")
    st.markdown(
        "<p class=\"section-lead\">Unggah gambar tanaman untuk melihat prediksi penyakit dan rekomendasi tindakan.</p>",
        unsafe_allow_html=True,
    )
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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Label", prediction.get("predicted_label", "-"))
        col2.metric("Confidence", f"{prediction.get('confidence', 0.0):.2%}")
        col3.metric("Model", prediction.get("model_version", "-"))
        col4.metric("Inference Time", f"{prediction.get('inference_time_ms', 0):.2f} ms")

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
    st.markdown(
        "<p class=\"section-lead\">Ringkas performa model, confidence, dan antrian active learning.</p>",
        unsafe_allow_html=True,
    )
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
    st.markdown(
        "<p class=\"section-lead\">Validasi kasus berisiko tinggi untuk meningkatkan kualitas dataset.</p>",
        unsafe_allow_html=True,
    )
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

elif page == "Model Registry":
    st.header("Model Registry")
    st.markdown(
        "<p class=\"section-lead\">Lihat metadata model aktif dan riwayat training yang terdaftar.</p>",
        unsafe_allow_html=True,
    )
    metadata = api_get("/model/current")
    if metadata:
        current_version = metadata.get("current_model_version") or metadata.get("model_version")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Model Version", current_version)
        col2.metric("Num Classes", metadata.get("num_classes", "-"))
        col3.metric("Input Size", metadata.get("input_size", "-"))

        registry_response = api_get("/model/registry")
        registry_rows = registry_response.get("items", []) if registry_response else []
        if registry_rows:
            st.subheader("Registered Artifacts")
            columns = [
                "version",
                "status",
                "model_name",
                "accuracy",
                "macro_f1",
                "epochs",
                "batch_size",
                "learning_rate",
                "optimizer",
                "scheduler",
                "input_size",
                "num_classes",
                "train_samples",
                "val_samples",
                "test_samples",
                "training_platform",
                "training_device",
                "created_at",
                "notes",
            ]
            st.dataframe(pd.DataFrame(registry_rows).reindex(columns=columns), use_container_width=True)

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

elif page == "Tentang":
    st.header("Tentang")
    st.markdown(
        "<p class=\"section-lead\">Ringkasan tujuan, fitur, dan capaian utama AgriMLOps PlantWild.</p>",
        unsafe_allow_html=True,
    )

    about_cols = st.columns([0.6, 0.4])
    with about_cols[0]:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">Ringkasan Proyek</div>
                <div class="card-body">
                    AgriMLOps PlantWild adalah aplikasi web untuk diagnosis penyakit tanaman berbasis foto.
                    Sistem ini menghubungkan inferensi, pengumpulan feedback, monitoring, dan model registry
                    agar siklus perbaikan model lebih terarah.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style=\"height:12px\"></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">Fitur Utama</div>
                <div class="card-body">
                    <ul>
                        <li>Diagnosis cepat dari foto dengan confidence dan top-3.</li>
                        <li>Feedback user dan active learning queue untuk validasi.</li>
                        <li>Monitoring distribusi prediksi dan performa model.</li>
                        <li>Model registry untuk melacak artefak training.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with about_cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">Model Terkini (v3)</div>
                <div class="card-body">
                    Accuracy: <strong>0.8288</strong><br/>
                    Macro F1: <strong>0.8127</strong><br/>
                    Best Val Macro F1: <strong>0.8254</strong><br/>
                    Validated feedback: <strong>9</strong> sampel
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style=\"height:12px\"></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">Catatan Implementasi</div>
                <div class="card-body">
                    Feedback berasal dari controlled validated feedback simulation (bukan petani asli).
                    Retraining masih semi-manual menggunakan Kaggle GPU.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.warning("Halaman tidak ditemukan.")

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

st.markdown("<div class=\"footer-note\">AgriMLOps PlantWild - Finalist Showcase</div>", unsafe_allow_html=True)


