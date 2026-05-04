# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang

Pertanian merupakan salah satu sektor paling penting dalam perekonomian global, menyediakan pangan bagi populasi dunia yang terus bertambah. Namun, sektor ini menghadapi berbagai tantangan signifikan, salah satunya adalah serangan penyakit pada tanaman yang dapat menyebabkan kerugian hasil panen yang besar. Organisasi Pangan dan Pertanian (FAO) memperkirakan bahwa penyakit tanaman bertanggung jawab atas kehilangan hasil panen hingga 40% secara global setiap tahunnya (Shoaib et al., 2023). Kerugian ini tidak hanya berdampak pada ketahanan pangan tetapi juga pada ekonomi petani dan kesejahteraan masyarakat secara luas.

Secara tradisional, diagnosis penyakit tanaman dilakukan secara manual dengan mengamati gejala visual pada daun, batang, atau buah tanaman. Metode ini bergantung pada keahlian ahli pertanian atau penyuluh lapangan yang memiliki pengetahuan mendalam tentang berbagai jenis penyakit tanaman. Namun, pendekatan manual memiliki beberapa keterbatasan: membutuhkan waktu yang lama, subjektif, bergantung pada ketersediaan ahli, dan tidak dapat diimplementasikan secara skala besar (Upadhyay et al., 2025). Di wilayah dengan akses terbatas ke layanan ekstensi pertanian, petani seringkali tidak dapat mendapatkan diagnosis yang cepat dan akurat, sehingga penanganan penyakit menjadi terlambat dan kerugian menjadi lebih besar.

Perkembangan teknologi deep learning dan computer vision dalam dekade terakhir telah membuka peluang baru untuk diagnosis penyakit tanaman yang otomatis dan akurat. Deep learning, khususnya Convolutional Neural Networks (CNN), telah menunjukkan performa luar biasa dalam berbagai tugas klasifikasi gambar, termasuk deteksi penyakit tanaman (Shoaib et al., 2023). Berbagai studi telah melaporkan akurasi di atas 90% untuk deteksi penyakit tanaman menggunakan model deep learning yang dilatih pada dataset seperti PlantVillage (Wang et al., 2025). Namun, sebagian besar penelitian ini dilakukan dalam lingkungan terkontrol dengan gambar yang diambil di laboratorium atau kondisi yang ideal, yang tidak mencerminkan kondisi nyata di lapangan (in-the-wild).

Dataset yang digunakan dalam penelitian sebelumnya seringkali "menggampangkan" kondisi real-world dengan menggunakan gambar yang diambil dengan pencahayaan yang baik, latar belakang yang bersih, dan sudut pengambilan yang optimal (Xu et al., 2024). Dalam praktiknya, gambar yang diambil di lapangan memiliki variasi yang jauh lebih besar: pencahayaan yang tidak seragam, latar belakang yang kompleks, kondisi cuaca yang bervariasi, dan kualitas kamera yang berbeda-beda (Wei et al., 2024). Hal ini menyebabkan model yang dilatih pada dataset "bersih" sering gagal saat dideploy di lingkungan nyata, sebuah fenomena yang dikenal sebagai domain shift (Wu et al., 2023).

Selain tantangan dataset, implementasi sistem diagnosis penyakit tanaman berbasis deep learning dalam produksi juga menghadapi tantangan dalam pengelolaan siklus hidup model (model lifecycle). Model machine learning tidak hanya perlu dilatih dan dideploy sekali, tetapi perlu dipantau kinerjanya secara terus-menerus, diperbarui dengan data baru, dan dikelola versinya untuk memastikan kualitas prediksi tetap optimal (Behl, 2025). Pendekatan tradisional yang hanya fokus pada pelatihan model tanpa mempertimbangkan aspek operasional seringkali gagal dalam produksi karena model mengalami degradation, drift, atau tidak dapat beradaptasi dengan kondisi baru.

Machine Learning Operations (MLOps) muncul sebagai disiplin baru yang menggabungkan praktik DevOps dengan machine learning untuk mengelola siklus hidup model secara end-to-end (Behl, 2025). MLOps mencakup aspek seperti continuous integration dan deployment (CI/CD), monitoring model, deteksi drift, model registry, dan automated retraining. Implementasi MLOps memungkinkan model machine learning untuk dideploy, dipantau, dan diperbarui secara sistematis, memastikan kinerja model tetap optimal dalam jangka panjang.

Konsep active learning merupakan komponen penting dalam MLOps untuk sistem yang beroperasi di lingkungan yang dinamis. Active learning memungkinkan sistem untuk secara selektif meminta label pada data yang paling informatif, seperti prediksi dengan confidence rendah, sehingga meningkatkan efisiensi proses labeling (Dong et al., 2023). Dalam konteks diagnosis penyakit tanaman, active learning dapat digunakan untuk mengidentifikasi prediksi yang tidak pasti dan meminta validasi dari ahli pertanian, kemudian menggunakan data yang divalidasi tersebut untuk retraining model secara iteratif. Pendekatan ini tidak hanya meningkatkan akurasi model tetapi juga mengurangi biaya labeling dengan fokus pada data yang paling bernilai.

Dataset PlantWild (Wei et al., 2024) merupakan dataset in-the-wild yang menyediakan gambar dan deskripsi teks untuk berbagai penyakit tanaman yang diambil dalam kondisi lapangan nyata. Dataset ini menyoroti tantangan small inter-class discrepancy dan large intra-class variance yang umum di lapangan, di mana gambar dari kelas yang sama dapat terlihat sangat berbeda karena variasi kondisi, sementara gambar dari kelas berbeda dapat terlihat mirip. PlantWild menyediakan baseline yang realistis untuk mengembangkan dan menguji sistem diagnosis penyakit tanaman yang robust terhadap kondisi lapangan.

Model EfficientNetV2-B0 telah terbukti memberikan keseimbangan yang baik antara akurasi dan efisiensi komputasi, menjadikannya cocok untuk deployment pada berbagai platform termasuk mobile dan edge devices (Upadhyay et al., 2025). Model ini dapat dilatih menggunakan transfer learning dari ImageNet, mengurangi waktu dan sumber daya yang dibutuhkan untuk pelatihan sambil tetap mencapai performa yang kompetitif.

Deployment sistem pada cloud infrastructure seperti DigitalOcean Droplet dengan Docker Compose memungkinkan skalabilitas dan aksesibilitas yang mudah. Docker containerization memastikan konsistensi lingkungan antara development dan production, sementara cloud deployment menyediakan akses global dan kemudahan pemeliharaan (Behl, 2025).

Berdasarkan latar belakang tersebut, penelitian ini bertujuan untuk mengimplementasikan sistem diagnosis penyakit tanaman berbasis deep learning dengan mengintegrasikan MLOps lifecycle dan active learning untuk pengelolaan model yang berkelanjutan, serta mendeploy sistem tersebut pada cloud infrastructure untuk akses yang luas.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini adalah:

1. Bagaimana mengimplementasikan sistem diagnosis penyakit tanaman menggunakan deep learning (EfficientNetV2-B0) dengan dataset in-the-wild (PlantWild)?
2. Bagaimana mengintegrasikan Machine Learning Operations (MLOps) lifecycle yang mencakup inference, logging, feedback collection, active learning queue, monitoring, dan model registry untuk pengelolaan model yang berkelanjutan?
3. Bagaimana menerapkan active learning untuk peningkatan kualitas model secara iteratif melalui validasi prediksi yang tidak pasti?
4. Bagaimana mendeploy sistem pada cloud infrastructure (DigitalOcean Droplet) dengan Docker Compose untuk akses yang luas dan skalabilitas?

## 1.3 Tujuan Penelitian

Tujuan penelitian ini adalah:

1. Mengimplementasikan sistem diagnosis penyakit tanaman menggunakan EfficientNetV2-B0 dengan dataset PlantWild subset (15 kelas) untuk klasifikasi penyakit tanaman in-the-wild.
2. Mengintegrasikan MLOps lifecycle lengkap yang mencakup: (a) model inference dengan FastAPI, (b) prediction logging dengan SQLite, (c) user feedback collection, (d) active learning queue untuk validasi prediksi, (e) monitoring dashboard, dan (f) model registry untuk version management.
3. Menerapkan active learning queue untuk mengidentifikasi dan memvalidasi prediksi dengan confidence rendah atau user unsure, kemudian menggunakan data yang divalidasi untuk retraining model secara iteratif.
4. Mendeploy sistem pada DigitalOcean Droplet dengan Docker Compose untuk menyediakan akses web (Streamlit) dan API (FastAPI) yang skalabel dan mudah diakses.

## 1.4 Manfaat Penelitian

### 1.4.1 Manfaat Teoretis

Penelitian ini memberikan kontribusi teoretis sebagai berikut:

1. **Kontribusi pada Literatur MLOps untuk Aplikasi Pertanian**: Mengisi gap literatur dengan mendemonstrasikan implementasi end-to-end MLOps lifecycle khusus untuk diagnosis penyakit tanaman, yang sebelumnya belum banyak dieksplorasi secara komprehensif.
2. **Demonstrasi Active Learning dalam Konteks Nyata**: Menunjukkan bagaimana active learning dapat diterapkan dalam sistem diagnosis penyakit tanaman untuk peningkatan kualitas model secara iteratif dengan efisiensi labeling yang optimal.
3. **Studi Kasus In-the-Wild Deployment**: Memberikan studi kasus praktis tentang tantangan dan solusi dalam deployment model deep learning untuk diagnosis penyakit tanaman dalam kondisi lapangan nyata.

### 1.4.2 Manfaat Praktis

Penelitian ini memberikan manfaat praktis sebagai berikut:

1. **Sistem Diagnosis Penyakit Tanaman yang Dapat Digunakan**: Menghasilkan sistem diagnosis penyakit tanaman yang dapat digunakan secara langsung oleh petani atau praktisi pertanian melalui antarmuka web yang mudah diakses.
2. **Template MLOps untuk Proyek Machine Learning Lainnya**: Menyediakan template dan best practices implementasi MLOps yang dapat diadaptasi untuk proyek machine learning lainnya di bidang pertanian atau domain lain.
3. **Panduan Deployment Cloud untuk Aplikasi AI**: Memberikan panduan praktis tentang deployment aplikasi AI pada cloud infrastructure dengan Docker Compose, yang dapat digunakan sebagai referensi untuk deployment sistem serupa.
4. **Model Registry dan Version Management**: Menunjukkan praktik terbaik dalam pengelolaan model versioning dan comparison, memudahkan tracking dan rollback jika diperlukan.
5. **Monitoring dan Observability**: Mendemonstrasikan implementasi monitoring dashboard untuk tracking kinerja model, feedback distribution, dan sistem health secara real-time.
