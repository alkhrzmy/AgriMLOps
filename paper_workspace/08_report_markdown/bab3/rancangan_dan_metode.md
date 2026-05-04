# BAB III
# RANCANGAN DAN METODE IMPLEMENTASI

## 3.1 Pendekatan Pemecahan Masalah

Penelitian ini mengadopsi pendekatan berbasis MLOps lifecycle dengan iterative model development untuk mengimplementasikan sistem diagnosis penyakit tanaman yang berkelanjutan. Pendekatan ini terdiri dari beberapa iterasi pengembangan model (v1, v2, v3) dengan peningkatan bertahap melalui feedback loop dan active learning.

### 3.1.1 Iterasi Model Development

**Iterasi 1 - Baseline Model (v1)**:
- Training model EfficientNetV2-B0 dari ImageNet pretrained weights
- Dataset: PlantWild subset (15 kelas, 5.645 gambar)
- Tujuan: Membangun baseline untuk diagnosis penyakit tanaman
- Hasil: Accuracy 0.8217, Macro F1 0.8086

**Iterasi 2 - MLOps Integration**:
- Implementasi SQLite database untuk prediction logging, feedback collection, active learning queue, dan model registry
- Implementasi FastAPI endpoints untuk inference dan feedback
- Implementasi Streamlit UI untuk diagnosis, feedback, monitoring, dan active learning queue
- Tujuan: Mengintegrasikan MLOps lifecycle lengkap
- Hasil: Sistem end-to-end dengan feedback loop

**Iterasi 3 - Controlled Active Learning Simulation**:
- Simulasi feedback dari labeled dataset untuk menguji active learning workflow
- Validasi active learning queue items
- Export validated feedback untuk retraining
- Tujuan: Menguji active learning dan retraining workflow
- Hasil: 9 validated feedback samples

**Iterasi 4 - Model Retraining (v2)**:
- Retraining model dengan feedback (feedback_samples_used=0)
- Dataset: PlantWild subset (15 kelas, 5.645 gambar)
- Tujuan: Menguji retraining pipeline
- Hasil: Accuracy 0.8253, Macro F1 0.8124

**Iterasi 5 - Active Learning Retraining (v3)**:
- Retraining model dengan 9 validated feedback samples
- Transfer learning dari model v2
- Dataset: PlantWild subset + 9 validated samples
- Tujuan: Meningkatkan model dengan active learning
- Hasil: Accuracy 0.8288, Macro F1 0.8127

**Iterasi 6 - Production Deployment**:
- Deployment pada DigitalOcean Droplet dengan Docker Compose
- Model v3 sebagai current model
- Tujuan: Deployment ke produksi
- Hasil: Sistem berjalan di http://159.65.139.148:8000 (API) dan http://159.65.139.148:8501 (Web)

### 3.1.2 Continuous Improvement Loop

Sistem dirancang dengan continuous improvement loop yang mencakup:

1. **Inference**: User upload gambar melalui Streamlit UI → API melakukan prediksi → Hasil ditampilkan
2. **Logging**: Prediksi dilog ke database (prediction_logs)
3. **Feedback**: User memberikan feedback (correct/incorrect/unsure) → Feedback dilog (feedback_logs)
4. **Active Learning**: Prediksi dengan confidence < 0.70 atau user unsure ditambahkan ke queue (active_learning_queue)
5. **Validation**: Ahli validasi queue items → Validated label disimpan
6. **Retraining**: Validated feedback digunakan untuk retraining model
7. **Deployment**: Model baru dideploy setelah verifikasi
8. **Monitoring**: Kinerja model dipantau secara real-time

## 3.2 Konsep dan Spesifikasi Rancangan

### 3.2.1 Arsitektur Sistem

Sistem AgriMLOps terdiri dari tiga komponen utama:

1. **Backend API (FastAPI)**:
   - Menyediakan endpoint untuk inference, feedback, monitoring, dan active learning
   - Mengelola database SQLite untuk logging dan tracking
   - Mengimplementasikan model inference dengan PyTorch + timm
   - Port: 8000

2. **Frontend Web (Streamlit)**:
   - Menyediakan user interface untuk diagnosis, feedback, monitoring, dan active learning queue
   - Berkomunikasi dengan backend API melalui HTTP requests
   - Port: 8501

3. **Database (SQLite)**:
   - Menyimpan prediction logs, feedback logs, active learning queue, dan model registry
   - File: data/agrimlops.db

![Gambar 3.1: Arsitektur sistem AgriMLOps](images/gambar_3_1_system_architecture.png)

**Gambar 3.1: Arsitektur sistem AgriMLOps**

Sumber: Hasil implementasi penelitian

### 3.2.2 Model Architecture

**Model**: EfficientNetV2-B0 (tf_efficientnetv2_b0)

**Spesifikasi**:
- Input size: 224x224 pixel
- Number of classes: 15 (top kelas dari PlantWild)
- Base model: ImageNet pretrained (v1, v2) atau model v2 (v3)
- Parameter: ~7.7 juta
- Model size: ~22.8 MB

**Training Configuration (v3)**:
- Epochs: 12
- Batch size: 32
- Learning rate: 3e-4
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Weight decay: 0.0001
- Split ratio: 70/15/15 (train/val/test)
- Data augmentation: RandomResizedCrop(224, scale=0.75-1.0), RandomHorizontalFlip, ColorJitter
- Mixed precision training: AMP (Automatic Mixed Precision)

### 3.2.3 Database Schema

Database SQLite dengan 4 tabel utama:

1. **prediction_logs**:
   - id (String, primary key)
   - image_path (String)
   - predicted_label (String)
   - confidence (Float)
   - top_k_predictions (JSON)
   - needs_review (Boolean)
   - timestamp (DateTime)

2. **feedback_logs**:
   - id (String, primary key)
   - prediction_id (String, foreign key)
   - feedback_type (String: correct/incorrect/unsure)
   - suggested_label (String, nullable)
   - confidence (Float)
   - timestamp (DateTime)

3. **active_learning_queue**:
   - id (String, primary key)
   - prediction_id (String, foreign key)
   - predicted_label (String)
   - confidence (Float)
   - reason (String: low_confidence/user_incorrect)
   - status (String: pending/validated/rejected)
   - validated_label (String, nullable)
   - validated_at (DateTime, nullable)
   - validator_note (String, nullable)

4. **model_registry**:
   - id (String, primary key)
   - model_version (String)
   - model_name (String)
   - artifact_path (String)
   - accuracy (Float, nullable)
   - macro_f1 (Float, nullable)
   - dataset_name (String, nullable)
   - num_classes (Integer, nullable)
   - input_size (Integer, nullable)
   - epochs (Integer, nullable)
   - batch_size (Integer, nullable)
   - learning_rate (Float, nullable)
   - optimizer (String, nullable)
   - scheduler (String, nullable)
   - train_samples (Integer, nullable)
   - val_samples (Integer, nullable)
   - test_samples (Integer, nullable)
   - feedback_samples_used (Integer, nullable)
   - training_platform (String, nullable)
   - training_device (String, nullable)
   - notes (String, nullable)
   - status (String: deployed/candidate)
   - created_at (DateTime)

![Gambar 3.2: Skema database](images/gambar_3_2_database_schema.png)

**Gambar 3.2: Skema database AgriMLOps**

Sumber: Hasil implementasi penelitian

### 3.2.4 API Endpoints

FastAPI menyediakan 8 endpoint utama:

1. **GET /health**: Health check API, mengembalikan status API, model loaded, current model version
2. **GET /model/current**: Mengembalikan informasi model saat ini dan metadata
3. **GET /model/registry**: Mengembalikan semua model yang terdaftar dengan status (deployed/candidate)
4. **POST /predict**: Upload gambar untuk prediksi, mengembalikan label, confidence, top-3 predictions, recommendation
5. **POST /feedback**: Submit user feedback (correct/incorrect/unsure) dengan optional suggested_label
6. **GET /monitoring/summary**: Mengembalikan statistik monitoring (total predictions, feedback distribution, model performance)
7. **GET /active-learning/queue**: Mengembalikan items active learning queue yang pending
8. **POST /active-learning/{item_id}/validate**: Validate active learning queue item dengan validated_label dan optional validator_note

![Gambar 3.3: Struktur API endpoints](images/gambar_3_3_api_endpoints.png)

**Gambar 3.3: Struktur API endpoints FastAPI**

Sumber: Hasil implementasi penelitian

### 3.2.5 Streamlit UI

Streamlit menyediakan 5 halaman utama:

1. **Diagnosis Page**: Upload gambar untuk prediksi, display hasil (label, confidence, top-3), show recommendation, provide feedback form
2. **Feedback Page**: View recent feedback submissions, filter by feedback type
3. **Monitoring Dashboard**: Total predictions count, feedback distribution, model performance metrics, confidence distribution
4. **Active Learning Queue**: List pending items, display images, validate items with label and notes, filter by status
5. **Model Registry**: View all registered models, display current deployed model, show model metadata and metrics, compare model versions

![Gambar 3.4: Struktur halaman Streamlit](images/gambar_3_4_streamlit_pages.png)

**Gambar 3.4: Struktur halaman Streamlit UI**

Sumber: Hasil implementasi penelitian

### 3.2.6 Deployment Architecture

Docker Compose dengan 2 service:

1. **API Service**:
   - Base image: python:3.11-slim
   - Dependencies: FastAPI, PyTorch (CPU-only), timm, SQLAlchemy
   - Environment variables: CURRENT_MODEL_VERSION, DATABASE_URL, LABEL_MAP_PATH, UPLOAD_DIR
   - Volumes: ./data, ./models, ./reports
   - Port: 8000

2. **Web Service**:
   - Base image: python:3.11-slim
   - Dependencies: Streamlit, requests, pandas, plotly
   - Environment variables: API_BASE_URL
   - Volumes: ./data (untuk akses gambar)
   - Port: 8501
   - Depends on: api

![Gambar 3.5: Arsitektur deployment Docker Compose](images/gambar_3_5_deployment_architecture.png)

**Gambar 3.5: Arsitektur deployment Docker Compose**

Sumber: Hasil implementasi penelitian

## 3.3 Alat dan Bahan

### 3.3.1 Hardware

**Development**:
- Laptop untuk development dan testing

**Training**:
- Kaggle GPU (P100/T4) untuk training model
- Tidak menggunakan local CPU untuk training (terlalu lambat)

**Deployment**:
- DigitalOcean Droplet (Ubuntu)
- 1 vCPU, 1 GB RAM (basic plan)
- Storage: 25 GB SSD

### 3.3.2 Software

**Programming Language**:
- Python 3.11

**Backend Framework**:
- FastAPI 0.104.1
- Uvicorn 0.24.0 (ASGI server)
- SQLAlchemy 2.0.23 (ORM)
- Pydantic 2.5.0 (data validation)

**Frontend Framework**:
- Streamlit 1.28.1

**Machine Learning**:
- PyTorch 2.1.0 (CPU-only untuk deployment)
- timm 0.9.12 (PyTorch Image Models)
- torch torchvision (CPU-only dari PyTorch index)

**Dataset**:
- PlantWild (Hugging Face uqtwei2/PlantWild)
- 15 kelas top, 5.645 gambar total
- Stratified train/val/test split (70/15/15): 3.961 train, 847 val, 847 test

**Containerization**:
- Docker 24.0.7
- Docker Compose 2.21.0

**Version Control**:
- Git
- GitHub: https://github.com/alkhrzmy/AgriMLOps

**Tabel 3.1: Spesifikasi teknologi**

| Komponen | Teknologi | Fungsi |
|---|---|---|
| Backend | FastAPI | Inference, feedback, monitoring API |
| Frontend | Streamlit | UI diagnosis dan dashboard |
| Model | EfficientNetV2-B0 | Klasifikasi penyakit tanaman |
| Database | SQLite | Logs, feedback, queue, registry |
| Training | Kaggle GPU | Training/retraining model |
| Deployment | Docker Compose + DigitalOcean | Produksi web/API |

### 3.3.3 Dataset Preparation

**Source Dataset**: PlantWild (Hugging Face)

**Preprocessing**:
1. Download PlantWild dari Hugging Face
2. Extract dan scan gambar secara rekursif
3. Filter ke 15 kelas dengan jumlah gambar terbanyak
4. Stratified split 70/15/15 untuk train/val/test
5. Data augmentation untuk training:
   - Resize(256)
   - RandomResizedCrop(224, scale=0.75-1.0)
   - RandomHorizontalFlip
   - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
   - Normalize(ImageNet mean/std)

**Validated Feedback**:
- 9 samples dari controlled active learning simulation
- Format: ZIP dengan images/ dan metadata.csv
- metadata.csv berisi: image_path, label, validated_label

**Tabel 3.2: Dataset PlantWild subset**

| Aspek | Nilai |
|---|---|
| Sumber | Hugging Face uqtwei2/PlantWild |
| Jumlah kelas | 15 kelas top |
| Total gambar | 5.645 |
| Split | 70/15/15 |
| Train v1/v2 | 3.952 |
| Train v3 | 3.961, termasuk 9 controlled validated feedback samples |
| Val/Test | 847/847 |
| Input size | 224 x 224 |

## 3.4 Tahapan Implementasi Strategis

### 3.4.1 Tahap 1: Baseline Model (v1)

**Lokasi**: Kaggle GPU Notebook

**Script**: `notebooks/kaggle_train_plantwild.py`

**Proses**:
1. Download PlantWild dari Hugging Face
2. Scan dan filter ke 15 kelas top
3. Stratified split train/val/test
4. Load EfficientNetV2-B0 dari ImageNet pretrained
5. Replace final layer untuk 15 classes
6. Train selama 8 epochs dengan batch size 32
7. Evaluate pada test set
8. Export artifacts: model_v1.pt, model_v1_metadata.json, metrics_v1.json, classification_report_v1.csv, confusion_matrix_v1.png

**Hasil**:
- Training time: ~2 jam di Kaggle GPU
- Accuracy: 0.8217
- Macro F1: 0.8086
- Model size: 22.8 MB

### 3.4.2 Tahap 2: MLOps Integration

**Lokasi**: Local development

**Database Implementation**:
- Implementasi SQLite schema dengan 4 tabel
- Fungsi database: init_db, log_prediction, log_feedback, add_to_active_learning_queue, get_active_learning_queue, validate_active_learning_item, register_current_model_from_metadata, get_monitoring_summary

**API Implementation**:
- Implementasi 8 FastAPI endpoints
- Model inference dengan PyTorch + timm
- Database operations dengan SQLAlchemy
- Error handling dan validation

**Web Implementation**:
- Implementasi 5 halaman Streamlit
- Integration dengan API endpoints
- Image upload dan display
- Feedback form
- Monitoring dashboard dengan Plotly

**Docker Implementation**:
- Dockerfile.api (CPU-only PyTorch)
- Dockerfile.web (tanpa PyTorch, hanya Streamlit)
- docker-compose.yml dengan 2 service
- Volume mount untuk persistensi

### 3.4.3 Tahap 3: Controlled Active Learning Simulation

**Lokasi**: Local development

**Script**: `scripts/simulate_validated_feedback.py`

**Dataset Preparation**:
- Download PlantWild subset (5 gambar per kelas × 15 kelas = 75 gambar)
- Format: dataset_root/class_name/image.jpg

**Simulation Process**:
1. Load label_map.json (15 kelas)
2. Scan dataset dan filter ke kelas yang ada di label_map
3. Untuk setiap gambar:
   - Kirim POST /predict ke production API
   - Submit feedback (correct/incorrect/unsure)
   - Jika incorrect/unsure, validate matching pending queue items
4. Generate report: reports/controlled_feedback_simulation.json
5. Export validated feedback: data/retraining/validated_feedback_YYYYMMDD_HHMMSS.zip

**Hasil**:
- Total predictions: 75
- Feedback submitted: 75
- Validated queue items: 9
- Export: validated_feedback_20260504_025418.zip

![Gambar 3.7: Workflow controlled validated feedback simulation](images/gambar_3_7_controlled_feedback_simulation.png)

**Gambar 3.7: Workflow controlled validated feedback simulation**

Sumber: Hasil implementasi penelitian

Simulasi ini digunakan untuk menguji alur active learning secara terkendali. Feedback yang digunakan bukan berasal dari petani asli, melainkan dari data berlabel yang dipakai sebagai kontrol validasi. Oleh karena itu, hasil simulasi tidak diklaim sebagai validasi lapangan, tetapi sebagai bukti bahwa pipeline feedback, validasi, export, retraining, dan registry dapat berjalan.

### 3.4.4 Tahap 4: Model Retraining (v2)

**Lokasi**: Kaggle GPU Notebook

**Script**: `notebooks/kaggle_train_plantwild.py` dengan MODEL_VERSION='v2'

**Proses**:
1. Download PlantWild dari Hugging Face
2. Scan dan filter ke 15 kelas top
3. Stratified split train/val/test
4. Load EfficientNetV2-B0 dari ImageNet pretrained
5. Train selama 8 epochs dengan batch size 32
6. Evaluate pada test set
7. Export artifacts: model_v2.pt, model_v2_metadata.json, metrics_v2.json, classification_report_v2.csv, confusion_matrix_v2.png

**Hasil**:
- Accuracy: 0.8253 (+0.0036 vs v1)
- Macro F1: 0.8124 (+0.0038 vs v1)
- Feedback samples used: 0 (tidak ada feedback yang diintegrasikan)

### 3.4.5 Tahap 5: Active Learning Retraining (v3)

**Lokasi**: Kaggle GPU Notebook

**Script**: `notebooks/kaggle_train_plantwild.py` dengan MODEL_VERSION='v3', USE_VALIDATED_FEEDBACK=True

**Proses**:
1. Download PlantWild dari Hugging Face
2. Scan dan filter ke 15 kelas top
3. Stratified split train/val/test
4. Load model_v2.pt sebagai base model (transfer learning)
5. Extract validated feedback ZIP
6. Tambahkan 9 validated samples ke training set
7. Train selama 12 epochs dengan batch size 32
8. Evaluate pada test set
9. Export artifacts: model_v3.pt, model_v3_metadata.json, metrics_v3.json, classification_report_v3.csv, confusion_matrix_v3.png

**Hasil**:
- Accuracy: 0.8288 (+0.0035 vs v2)
- Macro F1: 0.8127 (+0.0003 vs v2)
- Best Val F1: 0.8254 (+0.0147 vs v2)
- Feedback samples used: 9

**Tabel 3.3: Konfigurasi training v1-v2-v3**

| Versi | Base model | Epoch | Batch | LR | Feedback | Platform |
|---|---|---:|---:|---:|---:|---|
| v1 | ImageNet | 8 | 32 | 0.0003 | 0 | Kaggle GPU |
| v2 | ImageNet | 8 | 32 | 0.0003 | 0 | Kaggle GPU |
| v3 | model_v2.pt | 12 | 32 | 0.0003 | 9 | Kaggle GPU |

**Tabel 3.4: Database schema**

| Tabel | Field utama | Fungsi |
|---|---|---|
| prediction_logs | id, image_path, predicted_label, confidence, top_k_predictions, needs_review, timestamp | Mencatat seluruh prediksi |
| feedback_logs | id, prediction_id, feedback_type, suggested_label, confidence, timestamp | Mencatat feedback user/evaluator |
| active_learning_queue | id, prediction_id, reason, status, validated_label | Queue validasi active learning |
| model_registry | model_version, artifact_path, accuracy, macro_f1, feedback_samples_used, status | Registry dan lineage model |

**Tabel 3.5: API endpoints**

| Endpoint | Method | Fungsi |
|---|---|---|
| /health | GET | Health check API dan model |
| /model/current | GET | Metadata model aktif |
| /model/registry | GET | Daftar versi model |
| /predict | POST | Prediksi gambar |
| /feedback | POST | Simpan feedback |
| /monitoring/summary | GET | Ringkasan monitoring |
| /active-learning/queue | GET | Daftar queue review |
| /active-learning/{item_id}/validate | POST | Validasi queue item |

![Gambar 3.6: Workflow training Kaggle](images/gambar_3_6_training_workflow.png)

**Gambar 3.6: Workflow training model di Kaggle**

Sumber: Hasil implementasi penelitian

### 3.4.6 Tahap 6: Production Deployment

**Lokasi**: DigitalOcean Droplet

**Proses**:
1. SSH ke Droplet (159.65.139.148)
2. Clone repository: git clone https://github.com/alkhrzmy/AgriMLOps
3. Copy artifacts v3 ke models/ dan reports/
4. Update .env: CURRENT_MODEL_VERSION=v3
5. Run: docker-compose down
6. Run: docker-compose up -d --build
7. Register model v3 di database
8. Verify deployment: GET /health, GET /model/current

**Hasil**:
- API: http://159.65.139.148:8000 ✓
- Web: http://159.65.139.148:8501 ✓
- Current model: v3 ✓
- All services running ✓

## 3.5 Analisis Kelayakan dan Evaluasi Dampak

### 3.5.1 Kelayakan Teknis

**Model Size**:
- model_v3.pt: 22.8 MB
- Di bawah GitHub 100 MB limit ✓
- Cocok untuk mobile deployment ✓
- Loading time: < 1 detik ✓

**Inference Time**:
- CPU inference: < 1 detik per gambar ✓
- API response time: < 500ms ✓
- Scalable dengan load balancing ✓

**Scalability**:
- Docker containerization memungkinkan horizontal scaling ✓
- Stateless API design ✓
- Database SQLite dapat diganti dengan PostgreSQL untuk scale lebih besar ✓

**Deployment Complexity**:
- Docker Compose menyederhanakan deployment ✓
- Environment consistency antara dev dan prod ✓
- Easy rollback dengan docker-compose down/up ✓

### 3.5.2 Kelayakan Ekonomis

**Cost Deployment**:
- DigitalOcean Droplet basic: ~$6/bulan ✓
- Kaggle GPU: Gratis untuk training ✓
- Total cost: ~$72/tahun ✓

**ROI Analysis**:
- Kerugian penyakit tanaman: ~$220 miliar/tahun global
- Potensi pengurangan kerugian dengan diagnosis cepat: 10-20%
- Biaya deployment sangat rendah dibanding potensi manfaat ✓
- ROI sangat tinggi untuk aplikasi pertanian ✓

**Maintenance Cost**:
- Minimal: hanya perlu monitoring dan retraining berkala
- Automated retraining dapat mengurangi biaya manual ✓

### 3.5.3 Dampak Sosial

**Manfaat untuk Petani**:
- Akses diagnosis cepat dan akurat ✓
- Mengurangi kerugian hasil panen ✓
- Tidak memerlukan keahlian khusus ✓
- Akses melalui web (mobile-friendly) ✓

**Dampak pada Ketahanan Pangan**:
- Mengurangi kehilangan hasil panen ✓
- Meningkatkan kualitas produk pertanian ✓
- Mendukung ketahanan pangan lokal ✓

**Dampak pada Lingkungan**:
- Penggunaan pestisida yang lebih tepat sasaran ✓
- Mengurangi overuse pestisida ✓
- Mendukung pertanian berkelanjutan ✓

### 3.5.4 Dampak Teknologi

**Inovasi**:
- Integrasi MLOps untuk aplikasi pertanian (relatif baru) ✓
- Active learning untuk diagnosis penyakit tanaman (novel) ✓
- Platform end-to-end untuk diagnosis penyakit tanaman (gap filling) ✓

**Replikabilitas**:
- Code open-source di GitHub ✓
- Docker containerization memudahkan deployment ✓
- Dokumentasi lengkap ✓
- Dapat diadaptasi untuk domain lain ✓

**Scalability**:
- Dapat di-scale ke lebih banyak kelas ✓
- Dapat diintegrasikan dengan IoT sensors ✓
- Dapat di-deploy pada edge devices ✓
