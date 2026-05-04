# BAB V
# PENUTUP

## 5.1 Kesimpulan

Berdasarkan hasil implementasi dan evaluasi, kesimpulan penelitian ini disusun sesuai dengan tujuan penelitian sebagai berikut.

1. Sistem diagnosis penyakit tanaman berbasis EfficientNetV2-B0 berhasil diimplementasikan pada PlantWild subset dengan 15 kelas dan 5.645 gambar. Model baseline v1 mencapai accuracy 0,8217 dan macro F1 0,8086, sedangkan model v3 mencapai accuracy 0,8288 dan macro F1 0,8127. Hasil ini menunjukkan bahwa model ringan berbasis transfer learning dapat menjadi dasar prototipe diagnosis citra tanaman in-the-wild.

2. Siklus MLOps berhasil diintegrasikan melalui FastAPI, Streamlit, SQLite, monitoring dashboard, dan model registry. Sistem mampu menjalankan inference, mencatat prediction_logs, menyimpan feedback_logs, menampilkan ringkasan monitoring, serta melacak metadata model v1, v2, dan v3. Integrasi ini membuat perkembangan model lebih terdokumentasi dan siap diperbarui.

3. Active learning queue berhasil diterapkan untuk mengelola prediksi yang perlu divalidasi. Controlled validated feedback simulation menghasilkan 9 sampel tervalidasi yang digunakan pada retraining model v3. Dibandingkan v2, model v3 meningkatkan best validation F1 dari 0,8107 menjadi 0,8254, meskipun peningkatan accuracy dan macro F1 masih kecil karena jumlah feedback masih terbatas.

## 5.2 Saran

Pengembangan berikutnya perlu memprioritaskan pengumpulan feedback dari pengguna lapangan atau ahli pertanian agar retraining tidak hanya bergantung pada simulasi. Proses validasi label juga perlu dibuat lebih ketat melalui role validator, riwayat keputusan, dan audit data sehingga kualitas feedback dapat dipertanggungjawabkan.

Sistem dapat dikembangkan menjadi aplikasi mobile atau Progressive Web App agar lebih mudah digunakan oleh petani di lapangan. Fitur tambahan seperti autentikasi, pembatasan akses admin, rate limiting, dan pencatatan audit perlu diterapkan sebelum sistem digunakan secara luas.

Dari sisi model, penelitian lanjutan dapat menambahkan Grad-CAM untuk menjelaskan area citra yang memengaruhi prediksi, memperluas kelas penyakit, dan menguji domain adaptation atau open-world detection. Pengujian usability dan evaluasi pada data lapangan Indonesia juga diperlukan agar manfaat sistem terhadap pengambilan keputusan pertanian dapat diukur secara lebih nyata.
