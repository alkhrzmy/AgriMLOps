# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang

Penyakit tanaman masih menjadi ancaman penting bagi ketahanan pangan karena dapat menurunkan hasil panen dan pendapatan petani. Shoaib et al. (2023) menjelaskan bahwa organisme pengganggu tanaman dapat menyebabkan kehilangan hasil yang besar, sedangkan Wei et al. (2024) menekankan bahwa pengenalan penyakit secara cepat diperlukan agar kerusakan tidak meluas. Kebutuhan tersebut sejalan dengan arah inovasi pertanian berbasis teknologi, yaitu menyediakan alat bantu diagnosis yang dapat dipakai secara lebih cepat, murah, dan konsisten oleh pengguna lapangan.

Diagnosis penyakit tanaman secara manual umumnya dilakukan melalui pengamatan gejala visual pada daun, batang, atau buah. Cara ini tetap penting, tetapi hasilnya bergantung pada ketersediaan ahli, pengalaman pengamat, dan kondisi lapangan. Rahman et al. (2025) juga mencatat bahwa pemeriksaan manual cenderung memakan waktu dan sulit diskalakan. Pada wilayah dengan akses penyuluh yang terbatas, keterlambatan diagnosis dapat membuat penanganan penyakit tidak tepat waktu.

Deep learning memberi peluang untuk mengenali pola visual penyakit tanaman secara otomatis. Berbagai penelitian menunjukkan bahwa CNN dan transfer learning mampu mencapai performa tinggi pada data citra tanaman (Shoaib et al., 2023; Upadhyay et al., 2025). Akan tetapi, performa yang baik pada dataset laboratorium belum selalu mencerminkan kondisi lapangan. Wei et al. (2024) menunjukkan bahwa citra in-the-wild memiliki latar belakang kompleks, pencahayaan beragam, sudut pengambilan tidak seragam, serta variasi gejala yang tinggi. Kondisi tersebut menyebabkan model lebih sulit melakukan generalisasi.

Tantangan berikutnya adalah pengelolaan model setelah sistem digunakan. Model pembelajaran mesin tidak cukup hanya dilatih sekali, karena distribusi data dapat berubah ketika sistem menerima citra baru dari lingkungan nyata. Behl (2025) menjelaskan bahwa MLOps dibutuhkan untuk menghubungkan pelatihan, deployment, monitoring, versioning, dan pembaruan model. Tanpa alur operasional tersebut, model yang awalnya baik dapat menurun kualitasnya dan sulit dilacak perubahannya.

Active learning dapat melengkapi MLOps karena sistem dapat memprioritaskan prediksi yang tidak pasti untuk divalidasi. Dong et al. (2023) menekankan pentingnya pembelajaran dinamis pada deteksi penyakit tanaman ketika sistem menghadapi data baru. Dalam penelitian ini, active learning tidak diklaim sebagai open-world detection penuh, tetapi digunakan sebagai mekanisme awal untuk mencatat prediksi ber-confidence rendah, meminta validasi label, dan menyiapkan data retraining.

Berdasarkan masalah tersebut, penelitian ini mengembangkan AgriMLOps sebagai prototipe sistem diagnosis penyakit tanaman berbasis EfficientNetV2-B0 pada dataset PlantWild. Fokus penelitian diarahkan pada tiga hal: implementasi model diagnosis, integrasi siklus MLOps, dan penerapan active learning queue untuk pembaruan model secara terkendali. Deployment pada server cloud digunakan sebagai bukti kelayakan akses sistem, tetapi bukan tujuan penelitian yang berdiri sendiri.

## 1.2 Rumusan Masalah

Rumusan masalah penelitian ini adalah sebagai berikut.

1. Bagaimana mengimplementasikan model deep learning untuk diagnosis penyakit tanaman pada citra in-the-wild menggunakan dataset PlantWild?
2. Bagaimana mengintegrasikan komponen MLOps, yaitu inference, prediction logging, feedback collection, monitoring, dan model registry, agar siklus hidup model dapat dilacak?
3. Bagaimana menerapkan active learning queue untuk memvalidasi prediksi yang tidak pasti dan menggunakan hasil validasi tersebut pada retraining model?

## 1.3 Tujuan Penelitian

Penelitian ini bertujuan menghasilkan prototipe AgriMLOps yang tidak hanya berfungsi sebagai model klasifikasi, tetapi juga sebagai sistem yang layak dikembangkan menuju alat bantu diagnosis bagi pengguna lapangan. Tujuan khusus penelitian ini adalah sebagai berikut.

1. Mengimplementasikan model EfficientNetV2-B0 untuk diagnosis penyakit tanaman berbasis citra PlantWild subset yang merepresentasikan kondisi in-the-wild.
2. Mengintegrasikan siklus MLOps melalui API inference, pencatatan prediksi, pengumpulan feedback, monitoring, dan model registry agar perkembangan model dapat dikelola secara sistematis.
3. Menerapkan active learning queue dan controlled validated feedback simulation untuk menguji alur validasi data, export feedback, retraining, dan perbandingan model v1 sampai v3.

## 1.4 Manfaat Penelitian

Secara teoretis, penelitian ini memberikan contoh implementasi MLOps pada domain diagnosis penyakit tanaman yang masih jarang dibahas sebagai platform end-to-end. Posisi penelitian ini melengkapi studi yang berfokus pada akurasi model, dataset in-the-wild, atau aplikasi web, dengan menambahkan logging, feedback, active learning queue, monitoring, dan model registry dalam satu alur kerja.

Secara praktis, prototipe ini dapat menjadi dasar pengembangan alat bantu diagnosis bagi petani atau penyuluh. Sistem web memungkinkan pengguna mengunggah citra tanaman, memperoleh prediksi, dan memberikan feedback. Walaupun feedback pada penelitian ini masih berupa simulasi tervalidasi, alur tersebut menunjukkan bagaimana sistem dapat disiapkan untuk menerima validasi ahli atau feedback pengguna lapangan pada tahap berikutnya.

Secara inovatif, AgriMLOps mendukung orientasi pertanian berbasis teknologi karena menggabungkan model ringan, dashboard monitoring, dan mekanisme perbaikan model. Pendekatan ini relevan dengan upaya pengurangan kehilangan hasil panen melalui diagnosis yang lebih cepat dan terdokumentasi.
