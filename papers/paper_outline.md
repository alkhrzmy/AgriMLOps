# Paper Outline - AgriMLOps PlantWild
# Format: Karya Tulis Ilmiah (KTI)

---

## COVER PAGE

**KARYA TULIS ILMIAH**

**IMPLEMENTASI SISTEM DIAGNOSA PENYAKIT TANAMAN DENGAN MACHINE LEARNING OPERATIONS (MLOPS) DAN ACTIVE LEARNING**

**SUBTEMA: SISTEM CERDAS BERBASIS DEEP LEARNING**

(Logo asal instalasi dicantumkan dengan ukuran 3x3)

**Asal Instansi**: INNOVEST
**Disusun Oleh**:
1. Nama Ketua
   NIM: [NIM]
2. Nama Anggota 1
   NIM: [NIM]
3. Nama Anggota 2
   NIM: [NIM]

**INNOVEST**
**JAKARTA**
**2026**

---

## KATA PENGANTAR

- Ucapan terima kasih
- Latar belakang penulisan
- Harapan penulis

---

## ABSTRAK

- Ringkasan penelitian (200-300 kata)
- Kata kunci: 5-7 kata

---

## DAFTAR ISI

- Semua bab dan sub-bab dengan halaman

---

## DAFTAR GAMBAR

- Semua gambar dengan keterangan dan halaman

---

## DAFTAR TABEL

- Semua tabel dengan keterangan dan halaman

---

## BAB I - PENDAHULUAN

### 1.1 Latar Belakang

- Pentingnya diagnosis penyakit tanaman pertanian
- Tantangan dalam deteksi penyakit tanaman secara manual
- Perkembangan teknologi deep learning untuk klasifikasi gambar
- Kebutuhan sistem MLOps untuk pengelolaan model machine learning
- Konsep active learning untuk peningkatan kualitas model
- Dataset PlantWild sebagai sumber data penyakit tanaman
- Model EfficientNetV2-B0 untuk klasifikasi gambar
- Deployment pada cloud infrastructure (DigitalOcean)

### 1.2 Rumusan Masalah

1. Bagaimana mengimplementasikan sistem diagnosis penyakit tanaman menggunakan deep learning?
2. Bagaimana mengintegrasikan MLOps lifecycle untuk pengelolaan model?
3. Bagaimana menerapkan active learning untuk peningkatan kualitas model?
4. Bagaimana mendeploy sistem pada cloud infrastructure?

### 1.3 Tujuan Penelitian

1. Mengimplementasikan sistem diagnosis penyakit tanaman menggunakan EfficientNetV2-B0
2. Mengintegrasikan MLOps lifecycle (inference, logging, feedback, retraining)
3. Menerapkan active learning queue untuk validasi prediksi
4. Mendeploy sistem pada DigitalOcean Droplet dengan Docker Compose

### 1.4 Manfaat Penelitian

**Manfaat Teoretis**:
- Kontribusi pada literatur MLOps untuk aplikasi pertanian
- Demonstrasi active learning dalam konteks nyata

**Manfaat Praktis**:
- Sistem diagnosis penyakit tanaman yang dapat digunakan
- Template MLOps untuk proyek machine learning lainnya
- Panduan deployment cloud untuk aplikasi AI

---

## BAB II - KAJIAN PUSTAKA

### 2.1 Penyakit Tanaman

- Jenis-jenis penyakit tanaman umum
- Dampak penyakit pada hasil pertanian
- Metode deteksi tradisional

### 2.2 Deep Learning untuk Klasifikasi Gambar

- Convolutional Neural Networks (CNN)
- Transfer learning
- EfficientNetV2 architecture

### 2.3 MLOps (Machine Learning Operations)

- Definisi MLOps
- Komponen MLOps lifecycle
- Best practices MLOps

### 2.4 Active Learning

- Konsep active learning
- Strategi active learning
- Aplikasi active learning dalam praktik

### 2.5 Model Registry dan Versioning

- Konsep model registry
- Versioning strategy
- Model comparison metrics

### 2.6 Deployment dan Containerization

- Docker dan Docker Compose
- Cloud deployment (DigitalOcean)
- Microservices architecture

### 2.7 Penelitian Terkait

- [Paper 1]: Plant disease detection using deep learning
- [Paper 2]: MLOps implementation in production
- [Paper 3]: Active learning for image classification
- [Paper 4]: EfficientNet applications in agriculture

---

## BAB III - RANCANGAN DAN METODE IMPLEMENTASI

### 3.1 Pendekatan Pemecahan Masalah

- Pendekatan berbasis MLOps lifecycle
- Iterative model development (v1, v2, v3)
- Continuous improvement dengan feedback loop

### 3.2 Konsep dan Spesifikasi Rancangan

**Arsitektur Sistem**:
- FastAPI backend
- Streamlit frontend
- SQLite database
- PyTorch + timm untuk inference
- Docker Compose untuk deployment

**Model Architecture**:
- EfficientNetV2-B0
- Input size: 224x224
- 15 classes (top dari PlantWild)
- Transfer learning dari ImageNet

**Database Schema**:
- prediction_logs
- feedback_logs
- active_learning_queue
- model_registry

### 3.3 Alat dan Bahan

**Hardware**:
- Laptop untuk development
- Kaggle GPU untuk training (P100/T4)
- DigitalOcean Droplet untuk deployment

**Software**:
- Python 3.11
- FastAPI, Streamlit
- PyTorch, timm
- SQLAlchemy
- Docker, Docker Compose

**Dataset**:
- PlantWild (Hugging Face uqtwei2/PlantWild)
- 15 classes, 5,645 images
- Stratified train/val/test split (70/15/15)

### 3.4 Tahapan Implementasi Strategis

**Tahap 1: Baseline Model (v1)**
- Training on Kaggle GPU
- 8 epochs, batch size 32
- Accuracy: 0.8217, Macro F1: 0.8086

**Tahap 2: MLOps Integration**
- Implementasi SQLite database
- API endpoints (/predict, /feedback, /monitoring)
- Streamlit UI (Diagnosis, Feedback, Monitoring)
- Active learning queue

**Tahap 3: Controlled Active Learning Simulation**
- Simulasi feedback dari labeled dataset
- Validasi active learning queue items
- Export validated feedback untuk retraining

**Tahap 4: Model Retraining (v2)**
- Training dengan feedback (feedback_samples_used=0)
- Accuracy: 0.8253, Macro F1: 0.8124

**Tahap 5: Active Learning Retraining (v3)**
- Training dengan 9 validated feedback samples
- Transfer learning dari v2
- 12 epochs
- Accuracy: 0.8288, Macro F1: 0.8127

**Tahap 6: Deployment**
- Docker containerization
- DigitalOcean Droplet deployment
- Model registry integration

### 3.5 Analisis Kelayakan dan Evaluasi Dampak

**Kelayakan Teknis**:
- Model size: 22.8 MB (di bawah GitHub limit)
- Inference time: < 1 detik per gambar
- Scalability dengan Docker

**Kelayakan Ekonomis**:
- Cost deployment: DigitalOcean Droplet (~$6/bulan)
- Kaggle GPU: Gratis untuk training
- ROI: Tinggi untuk aplikasi pertanian

**Dampak Sosial**:
- Membantu petani mendeteksi penyakit
- Mengurangi kerugian hasil panen
- Akses mudah melalui web interface

---

## BAB IV - HASIL DAN PEMBAHASAN

### 4.1 Hasil Pelatihan Model

**Model v1 (Baseline)**:
- Training time: ~2 jam di Kaggle GPU
- Accuracy: 0.8217
- Macro F1: 0.8086

**Model v2 (Retraining)**:
- Training time: ~2 jam di Kaggle GPU
- Accuracy: 0.8253 (+0.0036)
- Macro F1: 0.8124 (+0.0038)

**Model v3 (Active Learning)**:
- Training time: ~3 jam di Kaggle GPU
- Accuracy: 0.8288 (+0.0035)
- Macro F1: 0.8127 (+0.0003)
- Best Val F1: 0.8254 (+0.0147)

### 4.2 Implementasi MLOps Lifecycle

**Prediction Logging**:
- Semua prediksi tersimpan di database
- Metadata: label, confidence, timestamp

**Feedback Collection**:
- 75 predictions dengan feedback
- 9 validated items untuk retraining

**Active Learning Queue**:
- Otomatis tambah low confidence predictions
- Manual validation dengan image preview
- Status tracking (pending/validated/rejected)

**Model Registry**:
- Track v1, v2, v3
- Performance metrics comparison
- Deployment status management

### 4.3 Hasil Deployment

**DigitalOcean Droplet**:
- URL: http://159.65.139.148:8000 (API)
- URL: http://159.65.139.148:8501 (Web)
- Model v3 successfully deployed
- All services running

**Performance**:
- API response time: < 500ms
- Web UI responsive
- Database operations smooth

### 4.4 Pembahasan

**Improvement dari v1 ke v3**:
- Accuracy improvement: +0.0071
- Macro F1 improvement: +0.0041
- Best Val F1 improvement: +0.0147
- Active learning menunjukkan potensi improvement

**Tantangan**:
- Active learning queue issues (image display, validation)
- Database schema evolution
- Deployment complexity

**Solusi yang Diterapkan**:
- Volume mount untuk image access
- Auto-validation di feedback endpoint
- Batch validation script untuk retroactive fix

---

## BAB V - PENUTUP

### 5.1 Kesimpulan

1. Sistem diagnosis penyakit tanaman berhasil diimplementasikan dengan EfficientNetV2-B0
2. MLOps lifecycle terintegrasi penuh (inference, logging, feedback, retraining)
3. Active learning queue berhasil diimplementasikan dengan 9 validated samples
4. Sistem berhasil dideploy pada DigitalOcean Droplet
5. Model v3 menunjukkan improvement dibanding v1 dan v2

### 5.2 Saran

1. Integrasi dengan aplikasi mobile untuk petani
2. Implementasi real farmer feedback collection
3. Automated retraining pipeline dengan CI/CD
4. Penambahan fitur model explainability (Grad-CAM)
5. Implementasi authentication dan authorization
6. Multi-language support (Bahasa Indonesia)
7. Monitoring dengan Prometheus/Grafana
8. Model optimization untuk edge deployment

---

## DAFTAR PUSTAKA

[Daftar referensi dari papers yang didownload]

---

## LAMPIRAN

### Lampiran 1: Lembar Orisinalitas Karya

### Lampiran 2: Biodata Pelaksana

### Lampiran 3: Originalitas Turnitin

### Lampiran 4: Code Repository
- GitHub: https://github.com/alkhrzzy/AgriMLOps

### Lampiran 5: Model Metrics
- Confusion matrix v1, v2, v3
- Classification reports
- Training curves

### Lampiran 6: API Documentation
- Endpoint specifications
- Request/response examples

### Lampiran 7: Screenshots
- Streamlit UI pages
- Deployment status
