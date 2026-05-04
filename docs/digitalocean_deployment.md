# DigitalOcean Deployment Checklist

Deployment target menggunakan Docker Compose di DigitalOcean Droplet.

## Arsitektur Deployment Ringan

- **Kaggle GPU** menjalankan training dan retraining model.
- **GitHub** menyimpan source code dan model artifact final jika ukuran artifact masih sesuai.
- **DigitalOcean Droplet** hanya menjalankan inference API, Streamlit UI, SQLite DB, upload image storage, monitoring, dan active learning queue.
- Droplet tidak menginstall dependency training seperti `scikit-learn`, `matplotlib`, `seaborn`, atau `huggingface_hub`.

## Dependency Docker

- API container memakai `requirements-api.txt` dan PyTorch CPU-only wheel.
- Web container memakai `requirements-web.txt` dan tidak menginstall `torch`, `torchvision`, atau `timm`.
- Training Kaggle memakai `requirements-train.txt`.

## Why Training Is Not Done on the Droplet

- Training EfficientNetV2 membutuhkan GPU dan dependency berat.
- Droplet MVP dipakai untuk serving/inference agar hemat biaya.
- Retraining semi-manual dilakukan di Kaggle GPU dari validated feedback export.
- Ini sesuai praktik MLOps karena training pipeline dan serving infrastructure boleh dipisah.

## 1. Test Docker Lokal Terlebih Dahulu

```bash
python scripts/verify_artifacts.py
docker compose build api
docker compose build web
docker compose up --build
```

## 2. Buka Aplikasi Lokal

```text
http://localhost:8501
http://localhost:8000/docs
```

## 3. Test Flow Lokal

- Upload gambar di Streamlit.
- Submit feedback.
- Cek Dashboard Monitoring.
- Cek Active Learning Queue.
- Pastikan API docs bisa diakses.

## 4. Deploy ke DigitalOcean

```bash
ssh root@SERVER_IP
apt update
apt install -y docker.io docker-compose-plugin git
systemctl enable --now docker
git clone https://github.com/alkhrzmy/AgriMLOps.git
cd AgriMLOps
cp .env.example .env
echo "CURRENT_MODEL_VERSION=v1" >> .env
python3 scripts/verify_artifacts.py
docker compose up -d --build
```

## 5. Firewall Demo Minimal

```bash
ufw allow OpenSSH
ufw allow 8501/tcp
ufw allow 8000/tcp
ufw --force enable
```

## 6. Akses Demo

```text
http://SERVER_IP:8501
http://SERVER_IP:8000/docs
```

## 7. Catatan Keamanan

- Jangan expose halaman active learning/admin untuk publik tanpa proteksi.
- Untuk demo lomba boleh pakai IP sementara.
- Untuk produksi gunakan Nginx reverse proxy, HTTPS Certbot, dan auth.
- Batasi ukuran upload.
- Pastikan artifact model tersedia sebelum menjalankan container.
