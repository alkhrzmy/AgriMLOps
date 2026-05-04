# AgriMLOps PlantWild

AgriMLOps PlantWild adalah MVP web app diagnosis penyakit tanaman in-the-wild untuk lomba INNOVEST. Project ini menggabungkan inference model, logging prediksi, feedback user, active learning queue, monitoring sederhana, dan rancangan retraining agar model tidak berhenti hanya sebagai eksperimen notebook.

Training model berat dilakukan di Kaggle GPU. Repo lokal/Windsurf difokuskan untuk web app, API, MLOps loop, Docker, GitHub, dan deployment DigitalOcean.

## Konsep Produk

Alur utama platform:

1. User upload foto tanaman dari kondisi lapangan.
2. Backend melakukan prediksi penyakit menggunakan model klasifikasi berbasis PlantWild.
3. Aplikasi menampilkan label prediksi, confidence, top-k prediksi, dan rekomendasi tindakan.
4. Setiap prediksi disimpan ke SQLite sebagai log operasional model.
5. User memberi feedback apakah prediksi benar, salah, atau tidak yakin.
6. Prediksi low-confidence dan feedback bermasalah masuk ke active learning queue.
7. Admin atau validator memeriksa data di queue dan memberi label validasi.
8. Data tervalidasi dapat dipakai untuk retraining model versi berikutnya.
9. Dashboard monitoring menampilkan performa operasional, confidence, distribusi prediksi, dan indikasi drift sederhana.

## Stack

- **Backend**: FastAPI untuk inference API dan MLOps endpoints
- **Frontend**: Streamlit untuk UI diagnosis, monitoring, feedback, dan active learning
- **Database**: SQLite untuk prediction logs, feedback, model registry, dan active learning queue
- **ML Training**: PyTorch + timm di Kaggle GPU
- **Deployment**: Docker Compose di DigitalOcean Droplet untuk serving/inference saja

## Deployment Architecture

- **Kaggle GPU**: training dan retraining EfficientNetV2.
- **GitHub**: source code dan model artifact final jika ukuran masih layak untuk repository.
- **DigitalOcean Droplet**: inference API, Streamlit UI, SQLite DB, upload image storage, monitoring, active learning queue.
- **Droplet** tidak menjalankan training dan tidak membutuhkan dependency training seperti `scikit-learn`, `matplotlib`, `seaborn`, atau `huggingface_hub`.

## Dependency Files

- `requirements-api.txt`: dependency minimal FastAPI inference API.
- `requirements-web.txt`: dependency minimal Streamlit UI.
- `requirements-train.txt`: dependency training Kaggle.
- `requirements.txt`: compatibility untuk local development.

## Why Training Is Not Done on the Droplet

- Training EfficientNetV2 membutuhkan GPU dan dependency berat.
- Droplet MVP dipakai untuk serving/inference agar hemat biaya.
- Retraining semi-manual dilakukan di Kaggle GPU dari validated feedback export.
- Ini sesuai praktik MLOps karena training pipeline dan serving infrastructure boleh dipisah.

## Struktur Project

```text
agrimlops-plantwild/
├── app/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── services.py
│   └── web/
│       └── streamlit_app.py
├── src/
│   ├── data_download.py
│   ├── data_prepare.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── database.py
│   ├── monitoring.py
│   └── retrain.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── feedback/
├── notebooks/
│   └── kaggle_train_plantwild.py
├── scripts/
│   ├── export_kaggle_artifacts.py
│   └── verify_artifacts.py
├── artifacts_contract/
│   └── ARTIFACTS.md
├── docs/
│   ├── kaggle_training_workflow.md
│   ├── github_push_workflow.md
│   └── digitalocean_deployment.md
├── models/
├── reports/
├── docker/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.web
├── requirements.txt
├── .env.example
└── README.md
```

## Workflow Kaggle GPU

Alur kerja utama:

```text
Windsurf/local repo
→ buat pipeline dan notebook Kaggle
→ training di Kaggle GPU
→ download artifact model
→ masukkan artifact ke repo
→ verify artifact
→ push ke GitHub
→ deploy Docker Compose ke DigitalOcean
```

Langkah ringkas:

1. Buka Kaggle Notebook dan enable GPU.
2. Jalankan isi `notebooks/kaggle_train_plantwild.py`.
3. Download `/kaggle/working/agrimlops_artifacts.zip`.
4. Extract ZIP ke root repo lokal.
5. Jalankan:

```bash
python scripts/verify_artifacts.py
```

Artifact lengkap dijelaskan di `artifacts_contract/ARTIFACTS.md`.

## Artifact Model Wajib

Setelah training Kaggle, file berikut harus tersedia:

```text
models/model_v1.pt
models/label_map.json
models/model_v1_metadata.json
reports/metrics_v1.json
reports/confusion_matrix_v1.png
reports/classification_report_v1.csv
```

Jika `models/model_v1.pt` lebih dari 100MB, jangan commit ke GitHub. Gunakan GitHub Release, Hugging Face, atau object storage, lalu simpan URL sebagai `MODEL_URL`.

## Current Model Performance

Model production saat ini berasal dari training Kaggle GPU ketiga (v3) dengan 9 validated feedback samples pada subset 15 kelas PlantWild:

```text
model_version: v3
model_name: tf_efficientnetv2_b0
accuracy: 0.8288
macro_f1: 0.8127
input_size: 224
base_model: v2
feedback_samples_used: 9
artifact: models/model_v3.pt
```

## Menjalankan API Lokal

```bash
pip install -r requirements.txt
python scripts/verify_artifacts.py
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Health check tersedia di:

```text
http://localhost:8000/health
```

Contoh response setelah artifact tersedia:

```json
{
  "status": "ok",
  "service": "agrimlops-plantwild-api",
  "model_loaded": true,
  "model_version": "v1"
}
```

## Menjalankan Streamlit Lokal

```bash
streamlit run app/web/streamlit_app.py
```

Smoke test API:

```bash
python scripts/smoke_test_api.py
```

## Menjalankan dengan Docker Compose

```bash
python scripts/verify_artifacts.py
docker compose up --build
```

Akses aplikasi:

- Streamlit: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`

Current production model dipilih melalui environment variable:

```bash
CURRENT_MODEL_VERSION=v1
```

Jika model v2 sudah diverifikasi dan dipromosikan:

```bash
python scripts/promote_model.py --version v2
docker compose up -d --build
```

## Endpoint API

```text
GET  /health
GET  /model/current
GET  /model/registry
POST /predict
POST /feedback
GET  /monitoring/summary
GET  /active-learning/queue
POST /active-learning/{item_id}/validate
```

## Alur MLOps Produk

1. User upload foto tanaman melalui Streamlit.
2. Streamlit memanggil `POST /predict`.
3. FastAPI menyimpan upload ke `data/feedback/uploads/YYYYMMDD/`.
4. `src.predict` melakukan inference dengan artifact `models/model_v1.pt`.
5. Prediksi disimpan ke SQLite `data/agrimlops.db`.
6. Prediksi low-confidence masuk `active_learning_queue`.
7. User mengirim feedback benar/salah/tidak yakin melalui `POST /feedback`.
8. Feedback salah/tidak yakin masuk active learning queue.
9. Admin memvalidasi item queue di halaman Active Learning Queue.
10. Data tervalidasi dapat dipakai untuk retraining versi berikutnya di Kaggle GPU.

## Retraining Semi-Manual

Retraining tidak dijalankan lokal. Export feedback tervalidasi:

```bash
python scripts/export_validated_feedback.py
```

Setelah artifact v2 dari Kaggle diextract ke repo:

```bash
python scripts/verify_artifacts.py --version v2
python scripts/compare_models.py
python scripts/promote_model.py --version v2
```

Detail lengkap ada di `docs/retraining_workflow.md`.

## Roadmap Implementasi Bertahap

1. Siapkan script Kaggle untuk download PlantWild, subset dataset, training, evaluasi, dan export artifact.
2. Training EfficientNetV2-B0 baseline di Kaggle GPU.
3. Download dan verify artifact model di repo lokal.
4. Implementasi modul prediksi dan rekomendasi tindakan berbasis artifact.
5. Implementasi SQLite logging, feedback, active learning queue, dan model registry.
6. Implementasi endpoint FastAPI untuk prediksi, feedback, monitoring, active learning, dan model registry.
7. Implementasi UI Streamlit untuk diagnosis, dashboard monitoring, active learning, dan registry.
8. Tambahkan monitoring drift sederhana dan pipeline retraining semi-manual.
9. Deploy ke DigitalOcean Droplet menggunakan Docker Compose.
