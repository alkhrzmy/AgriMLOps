# BAB IV
# HASIL DAN PEMBAHASAN

## 4.1 Hasil Pelatihan Model

Pelatihan menghasilkan tiga versi model. Model v1 menjadi baseline dengan EfficientNetV2-B0 dan bobot awal ImageNet. Model v2 menguji pipeline retraining tanpa tambahan feedback tervalidasi. Model v3 menggunakan model v2 sebagai base model dan menambahkan 9 sampel controlled validated feedback ke data latih.

Model v1 mencapai accuracy 0,8217 dan macro F1 0,8086. Hasil ini menunjukkan bahwa transfer learning cukup efektif untuk membangun baseline pada PlantWild subset. Model v2 mencapai accuracy 0,8253 dan macro F1 0,8124. Peningkatan v2 terhadap v1 relatif kecil karena model v2 belum menggunakan feedback baru; metadata pelatihan menunjukkan feedback_samples_used bernilai 0.

Model v3 mencapai accuracy 0,8288, macro F1 0,8127, dan best validation F1 0,8254. Dibandingkan v2, peningkatan accuracy sebesar 0,0035 dan macro F1 sebesar 0,0003 masih kecil. Akan tetapi, best validation F1 meningkat 0,0147. Hasil ini menunjukkan bahwa penambahan feedback tervalidasi berpotensi memperbaiki generalisasi, meskipun jumlah feedback yang digunakan masih terbatas.

![Gambar 4.1: Perbandingan performa model v1, v2, v3](images/gambar_4_1_model_comparison.png)

**Gambar 4.1: Perbandingan performa model v1, v2, v3**

Sumber: Hasil implementasi penelitian

**Tabel 4.1: Perbandingan performa model**

| Metric | v1 | v2 | v3 | Δ (v3-v1) | Δ (v3-v2) |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.8217 | 0.8253 | 0.8288 | +0.0071 | +0.0035 |
| Macro F1 | 0.8086 | 0.8124 | 0.8127 | +0.0041 | +0.0003 |
| Best Val F1 | N/A | 0.8107 | 0.8254 | - | +0.0147 |
| Epochs | 8 | 8 | 12 | +4 | +4 |
| Feedback Samples | 0 | 0 | 9 | +9 | +9 |
| Base Model | ImageNet | ImageNet | v2 | - | Transfer learning |

Sumber: Hasil implementasi penelitian

Kurva pelatihan v3 menunjukkan bahwa model mencapai konvergensi setelah beberapa epoch awal dan nilai validasi relatif stabil pada akhir pelatihan. Tidak tampak indikasi overfitting yang kuat dari grafik yang dihasilkan. Interpretasi ini tetap perlu dibaca hati-hati karena ukuran feedback simulation masih kecil.

![Gambar 4.2: Training curves model v3](images/gambar_4_2_training_curves_v3.png)

**Gambar 4.2: Training curves model v3 (loss dan accuracy)**

Sumber: reports/metrics_v3.json

Confusion matrix dan contoh prediksi digunakan sebagai bukti evaluasi tambahan. Confusion matrix membantu melihat kelas yang masih berpotensi tertukar, sedangkan contoh prediksi menunjukkan bentuk output model pada beberapa citra uji.

![Gambar 4.3: Confusion matrix model v3](images/confusion_matrix_v3.png)

**Gambar 4.3: Confusion matrix model v3**

Sumber: reports/confusion_matrix_v3.png

![Gambar 4.4: Sample prediction model](images/sample_predictions.png)

**Gambar 4.4: Sample prediction model**

Sumber: reports/sample_predictions.png

## 4.2 Hasil Implementasi MLOps Lifecycle

Sistem berhasil mencatat prediksi ke tabel prediction_logs setiap kali endpoint /predict digunakan. Catatan tersebut memuat id prediksi, path gambar, label, confidence, top-k predictions, status needs_review, dan timestamp. Pencatatan ini penting karena prediksi produksi tidak hilang setelah hasil ditampilkan ke pengguna.

Feedback collection juga berjalan melalui endpoint /feedback dan halaman Streamlit. Feedback correct, incorrect, atau unsure disimpan ke feedback_logs. Jika feedback menunjukkan ketidakpastian atau kesalahan, sistem dapat mengaitkannya dengan active learning queue sehingga data tersebut dapat ditinjau ulang.

Active learning queue berfungsi sebagai ruang validasi untuk prediksi yang perlu diperiksa. Dalam simulasi terkendali, 75 citra berlabel dikirim ke API dan seluruhnya diberi feedback otomatis berdasarkan ground truth. Dari proses tersebut, 9 item divalidasi dan diekspor sebagai validated feedback untuk pelatihan model v3. Karena data berasal dari simulasi berbasis dataset berlabel, hasil ini tidak boleh diklaim sebagai feedback petani asli.

![Gambar 4.5: Prediction result pada Streamlit UI](images/gambar_4_6_prediction_result.png)

**Gambar 4.5: Prediction result pada Streamlit UI**

Sumber: Panel evidence production AgriMLOps dari http://159.65.139.148:8501

![Gambar 4.6: Active Learning Queue pada Streamlit UI](images/gambar_4_8_active_learning_queue.png)

**Gambar 4.6: Active Learning Queue pada Streamlit UI**

Sumber: Panel evidence production AgriMLOps dari http://159.65.139.148:8501

Monitoring dashboard menampilkan ringkasan jumlah prediksi, distribusi feedback, confidence, dan informasi model. Model registry menampilkan versi v1, v2, dan v3 beserta metadata utama. Dua komponen ini membuat perubahan model lebih mudah dilacak dibandingkan sistem prediksi tunggal tanpa registry.

![Gambar 4.7: Model Registry pada Streamlit UI](images/gambar_4_10_model_registry.png)

**Gambar 4.7: Model Registry pada Streamlit UI**

Sumber: Panel evidence production AgriMLOps dari http://159.65.139.148:8501

## 4.3 Pembahasan

Hasil penelitian menunjukkan bahwa integrasi MLOps memberi nilai tambah dibandingkan sekadar melatih model klasifikasi. Pada model v2, retraining tanpa feedback baru hanya memberi peningkatan kecil. Pada model v3, metadata menunjukkan feedback_samples_used berjumlah 9, sehingga gap pada v2 dapat ditutup secara lebih jelas. Walaupun peningkatan accuracy dan macro F1 masih kecil, alur validasi hingga retraining sudah terbukti berjalan.

Dari sisi inovasi, AgriMLOps menghubungkan diagnosis citra tanaman dengan siklus perbaikan model. Sistem tidak berhenti pada output prediksi, tetapi menyimpan riwayat, menerima feedback, memprioritaskan data tidak pasti, dan menampilkan registry model. Integrasi ini relevan dengan kebutuhan pertanian berbasis teknologi karena sistem dapat disiapkan untuk pembaruan berkelanjutan ketika data lapangan bertambah.

Keterbatasan utama penelitian adalah jumlah feedback tervalidasi masih sedikit dan berasal dari controlled simulation. Oleh sebab itu, model v3 belum dapat diklaim sebagai hasil pembelajaran dari feedback petani nyata. Penelitian ini juga belum menerapkan domain adaptation eksplisit, open-world detection penuh, authentication, atau pengujian usability dengan pengguna lapangan. Keterbatasan tersebut menjadi dasar saran pengembangan pada Bab V.

## 4.4 Hasil Deployment

Deployment dilakukan pada DigitalOcean Droplet menggunakan Docker Compose. Layanan API berjalan pada FastAPI, sedangkan layanan web berjalan pada Streamlit. Endpoint /health dan dokumentasi /docs dapat diakses untuk memeriksa status layanan. Deployment ini membuktikan bahwa prototipe dapat dijalankan sebagai web/API, tetapi pembahasan utama tetap berada pada implementasi model, MLOps lifecycle, dan active learning. Bukti visual deployment diringkas pada Gambar 4.5 sampai Gambar 4.7.
