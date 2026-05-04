# Analisis Visual Paper Referensi

Tujuan analisis ini bukan menyalin gambar paper, melainkan mengambil prinsip visual lalu menggambar ulang diagram AgriMLOps agar bebas plagiasi.

## Referensi yang dianalisis

1. **Wei et al. (2024) Snap and Diagnose**
   - Figure sistem menampilkan pipeline kiri-ke-kanan: input, encoder, retrieval/search, backend, database, dan UI.
   - Cocok sebagai inspirasi untuk `gambar_3_1_system_architecture.png` dan `gambar_3_5_deployment_architecture.png`.
   - Yang diterapkan: layout modular, blok komponen, dan alur panah. Konten diganti total sesuai AgriMLOps.

2. **Rahman et al. (2025) Real-time Monitoring System**
   - Figure arsitektur menunjukkan pemisahan training/evaluation, web/mobile application, dan service layer.
   - Cocok sebagai inspirasi untuk diagram deployment dan workflow inference.
   - Yang diterapkan: pemisahan UI, service, model, database, dan evaluation evidence.

3. **Behl (2025) MLOps Framework**
   - Visual yang tersedia berupa heatmap perbandingan tool dan pembahasan pipeline MLOps.
   - Cocok untuk prinsip lifecycle: CI/CD, monitoring, model versioning, drift, rollback/retraining.
   - Yang diterapkan: `gambar_2_3_mlops_lifecycle.png` digambar ulang sebagai loop AgriMLOps.

4. **Dong et al. (2023) Open-world Plant Disease Detection**
   - Figure paper menggunakan alur dynamic learning/update knowledge.
   - Cocok sebagai inspirasi konseptual untuk active learning, tetapi AgriMLOps tidak mengklaim true open-world detection.
   - Yang diterapkan: `gambar_2_4_active_learning_flow.png` hanya menggambarkan uncertainty/feedback/validation/retraining.

5. **Wei et al. (2024) PlantWild Benchmark**
   - Paper menekankan tantangan in-the-wild: small inter-class discrepancy dan large intra-class variance.
   - Cocok untuk narasi dataset dan research gap, bukan untuk disalin gambarnya.

## Keputusan desain ulang

- Semua diagram utama dibuat ulang dengan style konsisten: warna terbatas, rounded boxes, panah sederhana, font sans-serif jelas.
- Tidak ada gambar paper yang dicopy langsung ke laporan.
- Diagram yang menyebut “adaptasi” tetap berupa interpretasi konsep, bukan reproduksi visual.
- Screenshot production tidak lagi dipakai mentah penuh; dibuat panel evidence yang lebih rapi untuk DOCX.
