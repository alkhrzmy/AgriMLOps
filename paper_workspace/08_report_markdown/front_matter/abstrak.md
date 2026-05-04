# ABSTRAK

Penyakit tanaman dapat menurunkan produktivitas pertanian dan membutuhkan diagnosis yang cepat. Deep learning telah banyak digunakan untuk klasifikasi penyakit tanaman, tetapi sistem yang siap digunakan tidak hanya membutuhkan model akurat. Sistem juga perlu mencatat prediksi, menerima feedback, memantau performa, dan mengelola versi model agar dapat diperbarui ketika data baru tersedia.

Penelitian ini mengembangkan AgriMLOps, prototipe diagnosis penyakit tanaman berbasis EfficientNetV2-B0 pada dataset PlantWild subset yang merepresentasikan citra in-the-wild. Sistem mengintegrasikan FastAPI untuk inference, Streamlit untuk antarmuka web, SQLite untuk prediction logging dan feedback logging, active learning queue untuk validasi prediksi tidak pasti, monitoring dashboard, serta model registry. Pelatihan dilakukan dalam tiga versi: v1 sebagai baseline, v2 sebagai retraining tanpa feedback tervalidasi, dan v3 sebagai retraining berbasis controlled validated feedback simulation.

Model v3 mencapai accuracy 0,8288, macro F1 0,8127, dan best validation F1 0,8254. Nilai best validation F1 meningkat 0,0147 dibandingkan v2, sedangkan accuracy dan macro F1 meningkat kecil karena jumlah feedback tervalidasi masih 9 sampel. Hasil ini menunjukkan bahwa integrasi MLOps dan active learning queue dapat membuat alur diagnosis lebih terdokumentasi dan siap diperbarui. Feedback yang digunakan merupakan simulasi terkendali berbasis data berlabel, bukan feedback petani asli.

**Kata kunci**: MLOps, active learning, deep learning, penyakit tanaman, EfficientNetV2, PlantWild
