# Tabel API Endpoints

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
