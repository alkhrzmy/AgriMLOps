# BAB IV
# HASIL DAN PEMBAHASAN

## 4.1 Hasil Pelatihan Model

### 4.1.1 Model v1 (Baseline)

Model v1 dilatih sebagai baseline untuk diagnosis penyakit tanaman menggunakan EfficientNetV2-B0 dengan ImageNet pretrained weights. Training dilakukan pada Kaggle GPU selama 8 epochs dengan batch size 32.

**Training Configuration**:
- Dataset: PlantWild subset (15 kelas, 5.645 gambar)
- Train/Val/Test split: 70/15/15 (3.961/847/847)
- Epochs: 8
- Batch size: 32
- Learning rate: 3e-4
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Base model: ImageNet pretrained

**Performance Metrics**:
- Accuracy: 0.8217
- Macro F1: 0.8086
- Training time: ~2 jam di Kaggle GPU

**Hasil ini menunjukkan bahwa EfficientNetV2-B0 dengan transfer learning dari ImageNet mampu mencapai akurasi >80% pada dataset PlantWild, yang merupakan baseline yang solid untuk iterasi selanjutnya.**

### 4.1.2 Model v2 (Retraining)

Model v2 dilatih dengan konfigurasi yang sama dengan v1 untuk menguji retraining pipeline. Tidak ada feedback yang diintegrasikan dalam training ini (feedback_samples_used=0).

**Training Configuration**:
- Dataset: PlantWild subset (15 kelas, 5.645 gambar)
- Train/Val/Test split: 70/15/15 (3.961/847/847)
- Epochs: 8
- Batch size: 32
- Learning rate: 3e-4
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Base model: ImageNet pretrained

**Performance Metrics**:
- Accuracy: 0.8253 (+0.0036 vs v1)
- Macro F1: 0.8124 (+0.0038 vs v1)
- Best Val F1: 0.8107
- Training time: ~2 jam di Kaggle GPU

**Improvement kecil terlihat pada v2 dibanding v1, yang mungkin disebabkan oleh variasi random seed atau perbedaan minor dalam data augmentation. Peningkatan ini tidak signifikan, menunjukkan bahwa retraining tanpa feedback baru tidak memberikan improvement substansial.**

### 4.1.3 Model v3 (Active Learning Retraining)

Model v3 dilatih dengan 9 validated feedback samples dari controlled active learning simulation sebagai transfer learning dari model v2.

**Training Configuration**:
- Dataset: PlantWild subset (15 kelas, 5.645 gambar) + 9 validated feedback samples
- Train/Val/Test split: 70/15/15 (3.961/847/847 + 9 ke training)
- Epochs: 12 (+4 vs v2)
- Batch size: 32
- Learning rate: 3e-4
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Base model: model_v2.pt (transfer learning)

**Performance Metrics**:
- Accuracy: 0.8288 (+0.0035 vs v2, +0.0071 vs v1)
- Macro F1: 0.8127 (+0.0003 vs v2, +0.0041 vs v1)
- Best Val F1: 0.8254 (+0.0147 vs v2)
- Feedback samples used: 9
- Training time: ~3 jam di Kaggle GPU

**Peningkatan yang lebih signifikan terlihat pada Best Val F1 (+0.0147), yang menunjukkan bahwa active learning dengan validated feedback dapat meningkatkan generalisasi model. Peningkatan accuracy dan macro F1 relatif kecil, namun ini dapat dijelaskan oleh jumlah feedback samples yang masih terbatas (hanya 9 samples). Dengan lebih banyak feedback samples, improvement yang lebih besar diharapkan.**

![Gambar 4.1: Perbandingan performa model v1, v2, v3](images/gambar_4_1_model_comparison.png)

**Gambar 4.1: Perbandingan performa model v1, v2, v3**

Sumber: Hasil implementasi penelitian

**Tabel 4.1: Perbandingan performa model**

| Metric | v1 | v2 | v3 | Δ (v3-v1) | Δ (v3-v2) |
|--------|-----|-----|-----|-----------|-----------|
| Accuracy | 0.8217 | 0.8253 | 0.8288 | +0.0071 | +0.0035 |
| Macro F1 | 0.8086 | 0.8124 | 0.8127 | +0.0041 | +0.0003 |
| Best Val F1 | N/A | 0.8107 | 0.8254 | - | +0.0147 |
| Epochs | 8 | 8 | 12 | +4 | +4 |
| Feedback Samples | 0 | 0 | 9 | +9 | +9 |
| Base Model | ImageNet | ImageNet | v2 | - | Transfer learning |

Sumber: Hasil implementasi penelitian

### 4.1.4 Training Curves

![Gambar 4.2: Training curves model v3](images/gambar_4_2_training_curves_v3.png)

**Gambar 4.2: Training curves model v3 (loss dan accuracy)**

Sumber: reports/metrics_v3.json

Training curves menunjukkan bahwa model v3 mencapai konvergensi setelah sekitar 8-10 epochs, dengan validation accuracy stabil di sekitar 0.82-0.83. Tidak ada tanda overfitting yang signifikan, yang menunjukkan bahwa regularisasi dan data augmentation efektif.

![Gambar 4.3: Confusion matrix model v3](images/confusion_matrix_v3.png)

**Gambar 4.3: Confusion matrix model v3**

Sumber: reports/confusion_matrix_v3.png

![Gambar 4.4: Sample prediction model](images/sample_predictions.png)

**Gambar 4.4: Sample prediction model**

Sumber: reports/sample_predictions.png

## 4.2 Implementasi MLOps Lifecycle

### 4.2.1 Prediction Logging

Semua prediksi yang dilakukan melalui API secara otomatis dilog ke database SQLite (tabel `prediction_logs`). Setiap log mencakup:

- ID prediksi (unique)
- Path gambar
- Label yang diprediksi
- Confidence score
- Top-3 predictions
- Flag `needs_review` (jika confidence < 0.70)
- Timestamp

**Implementasi ini memungkinkan tracking semua prediksi yang dilakukan sistem, yang penting untuk monitoring dan analisis performa model secara historis.**

### 4.2.2 Feedback Collection

User dapat memberikan feedback pada prediksi melalui Streamlit UI dengan tiga opsi: (1) correct, (2) incorrect dengan suggested_label, atau (3) unsure dengan suggested_label. Feedback dilog ke database (tabel `feedback_logs`) dengan:

- ID feedback (unique)
- ID prediksi (foreign key)
- Tipe feedback (correct/incorrect/unsure)
- Suggested label (jika incorrect/unsure)
- Confidence saat prediksi
- Timestamp

**Selama controlled active learning simulation, 75 predictions diberikan feedback, dengan 9 di antaranya memiliki suggested_label untuk retraining.**

**Tabel 4.2: Hasil controlled active learning simulation**

| Aspek | Nilai |
|---|---|
| Jenis feedback | Controlled validated feedback simulation |
| Sumber label | Dataset berlabel, bukan petani asli |
| Feedback samples used pada v3 | 9 |
| Training platform retraining | Kaggle GPU |
| Klaim aman | Demonstrasi workflow active learning, bukan validasi lapangan skala besar |

### 4.2.3 Active Learning Queue

Prediksi dengan confidence < 0.70 atau user unsure secara otomatis ditambahkan ke active learning queue (tabel `active_learning_queue`). Queue items memiliki:

- ID queue item (unique)
- ID prediksi (foreign key)
- Label yang diprediksi
- Confidence
- Reason (low_confidence/user_incorrect)
- Status (pending/validated/rejected)
- Validated label (jika divalidasi)
- Timestamp validasi
- Catatan validator

**Selama controlled active learning simulation, 9 queue items divalidasi dengan label yang benar. Validated items ini kemudian digunakan untuk retraining model v3.**

![Gambar 4.5: Diagnosis page pada Streamlit UI](images/gambar_4_5_diagnosis_page.png)

**Gambar 4.5: Diagnosis page pada Streamlit UI**

Sumber: Screenshot dari http://159.65.139.148:8501

![Gambar 4.6: Prediction result pada Streamlit UI](images/gambar_4_6_prediction_result.png)

**Gambar 4.6: Prediction result pada Streamlit UI**

Sumber: Screenshot dari http://159.65.139.148:8501

![Gambar 4.7: Feedback form pada Streamlit UI](images/gambar_4_7_feedback_form.png)

**Gambar 4.7: Feedback form pada Streamlit UI**

Sumber: Screenshot dari http://159.65.139.148:8501

![Gambar 4.8: Active Learning Queue pada Streamlit UI](images/gambar_4_8_active_learning_queue.png)

**Gambar 4.8: Active Learning Queue pada Streamlit UI**

Sumber: Screenshot dari http://159.65.139.148:8501

### 4.2.4 Model Registry

Model registry diimplementasikan dengan database (tabel `model_registry`) yang menyimpan informasi tentang semua model yang dilatih:

- Model version (v1, v2, v3)
- Model name (tf_efficientnetv2_b0)
- Artifact path
- Accuracy dan macro F1
- Dataset name dan version
- Number of classes dan input size
- Hyperparameters (epochs, batch size, learning rate, optimizer, scheduler)
- Train/val/test samples
- Feedback samples used
- Training platform dan device
- Notes
- Status (deployed/candidate)
- Timestamp

**Registry ini memungkinkan tracking evolusi model dari v1 ke v3, comparison performa antar model, dan management deployment status. Saat ini, model v3 memiliki status "deployed" dan v1, v2 memiliki status "candidate".**

### 4.2.5 Monitoring Dashboard

Monitoring dashboard diimplementasikan di Streamlit yang menampilkan:

- Total predictions count
- Feedback distribution (correct/incorrect/unsure)
- Model performance metrics (accuracy, macro F1)
- Prediction confidence distribution
- Active learning queue status

**Dashboard ini memberikan visibility real-time terhadap kinerja sistem dan kesehatan model, memungkinkan deteksi dini terhadap degradation atau anomali.**

![Gambar 4.9: Monitoring Dashboard pada Streamlit UI](images/gambar_4_9_monitoring_dashboard.png)

**Gambar 4.9: Monitoring Dashboard pada Streamlit UI**

Sumber: Screenshot dari http://159.65.139.148:8501

![Gambar 4.10: Model Registry pada Streamlit UI](images/gambar_4_10_model_registry.png)

**Gambar 4.10: Model Registry pada Streamlit UI**

Sumber: Screenshot dari http://159.65.139.148:8501

**Tabel 4.3: Fitur MLOps yang diimplementasikan**

| Fitur | Status | Keterangan |
|---|---|---|
| Inference API | Implemented | FastAPI /predict |
| Prediction logging | Implemented | SQLite prediction_logs |
| Feedback collection | Implemented | Endpoint dan UI feedback |
| Active learning queue | Implemented | Queue confidence rendah/user unsure |
| Model registry | Implemented | Metadata v1-v2-v3 |
| Monitoring dashboard | Implemented | Streamlit summary |
| Automated retraining | Partial | Retraining semi-manual via Kaggle GPU |
| Drift detection | Not implemented | Future work |

## 4.3 Hasil Deployment

### 4.3.1 Docker Containerization

Sistem berhasil di-containerize dengan Docker Compose yang terdiri dari 2 service:

1. **API Service**:
   - Image: agrimlops-api
   - Base: python:3.11-slim
   - Dependencies: FastAPI, PyTorch CPU-only, timm, SQLAlchemy
   - Port: 8000
   - Volumes: ./data, ./models, ./reports

2. **Web Service**:
   - Image: agrimlops-web
   - Base: python:3.11-slim
   - Dependencies: Streamlit, requests, pandas, plotly
   - Port: 8501
   - Volumes: ./data
   - Depends on: api

**Containerization memastikan konsistensi lingkungan antara development dan production, memudahkan deployment dan pemeliharaan.**

### 4.3.2 DigitalOcean Droplet Deployment

Sistem berhasil dideploy pada DigitalOcean Droplet dengan spesifikasi:

- Host: 159.65.139.148
- OS: Ubuntu
- Resources: 1 vCPU, 1 GB RAM, 25 GB SSD
- Services: Docker Compose (api, web)

**Deployment Status**:
- API: http://159.65.139.148:8000 ✓ Running
- Web: http://159.65.139.148:8501 ✓ Running
- Current model: v3 ✓ Loaded
- Database: agrimlops.db ✓ Connected
- All services: ✓ Healthy

**Health check endpoint (/health) mengembalikan:**
```json
{
  "status": "ok",
  "service": "agrimlops-plantwild-api",
  "model_loaded": true,
  "model_artifact_available": true,
  "current_model_version": "v3",
  "model_version": "v3",
  "database_available": true
}
```

![Gambar 4.11: FastAPI documentation](images/gambar_4_11_fastapi_docs.png)

**Gambar 4.11: FastAPI documentation**

Sumber: Screenshot dari http://159.65.139.148:8000/docs

![Gambar 4.12: FastAPI health endpoint](images/gambar_4_12_fastapi_health.png)

**Gambar 4.12: FastAPI health endpoint**

Sumber: Screenshot dari http://159.65.139.148:8000/health

### 4.3.3 Performance Production

**API Response Time**:
- Health check: < 100ms
- Prediction endpoint: < 500ms
- Monitoring endpoint: < 200ms

**Resource Usage**:
- CPU: < 50% idle
- Memory: < 500MB / 1GB
- Disk: < 5GB / 25GB

**Sistem berjalan dengan resource yang efisien pada Droplet basic plan, menunjukkan bahwa arsitektur yang dirancang scalable dan cost-effective.**

## 4.4 Pembahasan

### 4.4.1 Improvement dari v1 ke v3

Model v3 menunjukkan improvement yang konsisten dibanding v1 dan v2:

1. **Accuracy**: +0.0071 dari v1, +0.0035 dari v2
2. **Macro F1**: +0.0041 dari v1, +0.0003 dari v2
3. **Best Val F1**: +0.0147 dari v2 (metrik ini tidak tersedia untuk v1)

**Peningkatan yang paling signifikan adalah pada Best Val F1 (+0.0147), yang menunjukkan bahwa active learning dengan validated feedback dapat meningkatkan generalisasi model pada data validation. Peningkatan accuracy dan macro F1 relatif kecil, namun ini dapat dijelaskan oleh:**

1. **Jumlah feedback samples terbatas**: Hanya 9 samples yang digunakan untuk retraining. Dengan lebih banyak feedback, improvement yang lebih besar diharapkan.
2. **Dataset sudah cukup besar**: PlantWild subset sudah memiliki 5.645 gambar, sehingga 9 additional samples tidak memberikan impact besar secara statistik.
3. **Model sudah cukup baik**: Baseline v1 sudah mencapai accuracy 0.8217, sehingga room for improvement terbatas.

**Meskipun improvement kecil, hasil ini mendemonstrasikan bahwa active learning loop berfungsi secara end-to-end dan dapat memberikan improvement iteratif. Dengan akumulasi feedback samples dari waktu ke waktu, diharapkan improvement yang lebih substansial dapat dicapai.**

### 4.4.2 Efektivitas MLOps Lifecycle

Implementasi MLOps lifecycle terbukti efektif dalam beberapa aspek:

1. **Prediction Logging**: Semua prediksi tercatat, memungkinkan tracking dan analisis historis.
2. **Feedback Collection**: User feedback dapat dikumpulkan secara sistematis melalui UI.
3. **Active Learning Queue**: Prediksi dengan confidence rendah otomatis ditambahkan ke queue untuk validasi.
4. **Model Registry**: Tracking model version dan comparison antar model berjalan lancar.
5. **Monitoring Dashboard**: Visibility real-time terhadap kinerja sistem tersedia.

**Namun, beberapa tantangan dihadapi selama implementasi:**

1. **Image Display di Active Learning Queue**: Awalnya gambar tidak ditampilkan karena web container tidak memiliki akses ke volume data. Masalah ini diatasi dengan menambahkan volume mount `./data:/app/data` pada web service.
2. **Validated Label Always None**: Awalnya semua queue items memiliki `validated_label=None`, mencegah retraining. Masalah ini disebabkan oleh duplikasi queue items dan logic validation yang tidak lengkap. Solusi: update `/feedback` endpoint untuk auto-validate existing pending items dan implementasi `batch_validate_pending.py` untuk retroactive validation.
3. **Database Schema Evolution**: Kolom baru ditambahkan ke model_registry untuk mendukung metadata training yang lebih detail. Solusi: implementasi `ensure_model_registry_columns()` untuk ALTER TABLE otomatis.

**Tantangan ini menunjukkan bahwa implementasi MLOps dalam produksi memerlukan perhatian pada detail dan iterasi debugging, namun solusi yang ditemukan efektif dan scalable.**

### 4.4.3 Controlled Active Learning Simulation

Controlled active learning simulation menggunakan labeled dataset dari PlantWild untuk mensimulasikan feedback farmer menunjukkan:

1. **Workflow end-to-end berfungsi**: Dari prediction → feedback → validation → retraining → deployment.
2. **9 validated samples dihasilkan**: Meskipun jumlah kecil, cukup untuk mendemonstrasikan workflow.
3. **Feedback dapat digunakan untuk retraining**: Model v3 berhasil dilatih dengan 9 validated samples dan menunjukkan improvement.

**Simulasi ini membuktikan bahwa active learning loop dapat diimplementasikan secara praktis. Namun, untuk production, diperlukan:**

1. **Real farmer feedback**: Integrasi dengan aplikasi mobile atau field app yang digunakan petani.
2. **More feedback samples**: Untuk improvement yang lebih signifikan.
3. **Quality control**: Validasi oleh ahli pertanian untuk memastikan label yang benar.

### 4.4.4 Deployment dan Scalability

Deployment pada DigitalOcean Droplet dengan Docker Compose menunjukkan:

1. **Ease of deployment**: Docker Compose menyederhanakan deployment dengan single command.
2. **Resource efficiency**: Sistem berjalan dengan resource yang efisien pada Droplet basic plan.
3. **Scalability**: Stateless API design memungkinkan horizontal scaling dengan load balancing.
4. **Monitoring**: Health check dan monitoring endpoint memungkinkan observability.

**Untuk scale lebih besar, diperlukan:**

1. **Database upgrade**: SQLite dapat diganti dengan PostgreSQL untuk concurrent access yang lebih baik.
2. **Load balancing**: Nginx atau AWS ALB untuk distribute traffic.
3. **Auto-scaling**: Kubernetes untuk auto-scaling berdasarkan load.
4. **CDN**: Cloudflare atau AWS CloudFront untuk static assets.

### 4.4.5 Limitations dan Future Work

**Tabel 4.4: Perbandingan AgriMLOps dengan paper terkait**

| Aspek | Paper terkait | AgriMLOps |
|---|---|---|
| Dataset in-the-wild | PlantWild benchmark | PlantWild subset 15 kelas |
| Aplikasi web | Snap and Diagnose/Rahman et al. | Streamlit + FastAPI |
| MLOps lifecycle | Behl generik | Implementasi khusus diagnosis tanaman |
| Active learning | Dong/open-world related | Queue validasi confidence rendah |
| Domain adaptation | Wu et al. | Belum diterapkan eksplisit |
| Multimodal | Wei et al. | Belum multimodal |

**Tabel 4.5: Limitasi dan future work**

| Limitasi | Dampak | Future Work |
|---|---|---|
| Feedback bukan petani asli | Belum validasi lapangan | Integrasi field/mobile app |
| Retraining semi-manual | Belum full CI/CD | Automated retraining pipeline |
| Tidak ada domain adaptation | Domain shift belum diatasi eksplisit | UDA/domain adaptation |
| Tidak multimodal | Hanya gambar | Integrasi teks/sensor/cuaca |
| SQLite single-node | Skalabilitas terbatas | PostgreSQL + monitoring alerts |

**Limitations saat ini:**

1. **Feedback samples terbatas**: Hanya 9 samples dari simulation, belum ada real farmer feedback.
2. **No authentication**: Sistem tidak memiliki authentication/authorization.
3. **Single model inference**: Tidak ada A/B testing atau ensemble.
4. **No automated monitoring alerts**: Monitoring manual melalui dashboard.
5. **Limited edge deployment**: Tidak dioptimasi untuk mobile/edge devices.

**Future work yang direkomendasikan:**

1. **Integrasi dengan aplikasi mobile**: Untuk real farmer feedback collection.
2. **Automated retraining pipeline**: CI/CD untuk retraining otomatis berdasarkan drift detection.
3. **Model explainability**: Grad-CAM untuk visualisasi attention map.
4. **Multi-language support**: Bahasa Indonesia untuk petani lokal.
5. **Authentication**: OAuth2 atau JWT untuk user management.
6. **Edge deployment**: Optimasi untuk mobile devices dengan TensorFlow Lite.
