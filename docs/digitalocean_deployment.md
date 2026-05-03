# DigitalOcean Deployment

Deployment target menggunakan Docker Compose di DigitalOcean Droplet.

## Setup Server

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git
sudo systemctl enable --now docker
git clone <GITHUB_REPO_URL>
cd agrimlops-plantwild
python3 scripts/verify_artifacts.py
sudo docker compose up -d --build
```

## Akses

```text
http://SERVER_IP:8501
http://SERVER_IP:8000/docs
```

## Catatan Produksi

- Gunakan Nginx reverse proxy.
- Gunakan domain dan HTTPS Certbot.
- Batasi ukuran upload.
- Jangan expose halaman admin tanpa autentikasi.
- Pastikan artifact model tersedia sebelum menjalankan container.
