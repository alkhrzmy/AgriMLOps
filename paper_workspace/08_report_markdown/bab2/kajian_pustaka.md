# BAB II
# KAJIAN PUSTAKA

## 2.1 Penyakit Tanaman

Penyakit tanaman merupakan salah satu ancaman terbesar dalam pertanian modern yang dapat menyebabkan kerugian ekonomi yang signifikan dan mengancam ketahanan pangan global. Menurut laporan Organisasi Pangan dan Pertanian (FAO), penyakit tanaman bertanggung jawab atas kehilangan hasil panen hingga 40% secara global setiap tahunnya, dengan estimasi kerugian ekonomi mencapai lebih dari $220 miliar (Shoaib et al., 2023). Kerugian ini tidak hanya berdampak pada petani skala kecil tetapi juga pada industri pertanian skala besar dan rantai pasok pangan secara keseluruhan.

Penyakit tanaman dapat diklasifikasikan berdasarkan penyebabnya menjadi tiga kategori utama: (1) penyakit biotik yang disebabkan oleh patogen seperti jamur, bakteri, virus, dan nematoda; (2) penyakit abiotik yang disebabkan oleh faktor lingkungan seperti kekurangan nutrisi, stres air, dan kondisi tanah yang tidak optimal; dan (3) penyakit fisiologis yang disebabkan oleh faktor genetik atau perkembangan tanaman (Wang et al., 2025). Penyakit biotik merupakan kategori yang paling umum dan paling merugikan, dengan jamur menjadi penyebab utama sekitar 70-80% dari semua penyakit tanaman (Shoaib et al., 2023).

![Gambar 2.1: Dampak penyakit tanaman pada hasil panen global](images/gambar_2_1_plant_disease_impact.png)

**Gambar 2.1: Dampak penyakit tanaman pada hasil panen global**

Sumber: Adaptasi dari Shoaib et al. (2023)

Di Indonesia, beberapa penyakit tanaman yang paling umum dan merugikan termasuk blast pada padi, hawar daun pada kentang, busuk buah pada tomat, dan karat pada gandum. Penyakit-penyakit ini tidak hanya mengurangi kuantitas hasil panen tetapi juga menurunkan kualitas produk pertanian, sehingga mengurangi nilai jual di pasar. Selain itu, penggunaan pestisida yang berlebihan untuk mengendalikan penyakit tanaman telah menimbulkan masalah lingkungan dan kesehatan, menekankan kebutuhan akan metode diagnosis yang lebih akurat dan tepat sasaran (Bouacida et al., 2024).

Metode tradisional diagnosis penyakit tanaman bergantung pada pengamatan visual manual oleh ahli pertanian atau penyuluh lapangan. Ahli mengidentifikasi penyakit berdasarkan gejala visual seperti perubahan warna daun, bercak, bercak busuk, atau deformasi pada bagian tanaman. Metode ini memiliki beberapa keterbatasan: (1) membutuhkan keahlian khusus yang tidak selalu tersedia di semua lokasi; (2) subjektif dan dapat bervariasi antar ahli; (3) memakan waktu dan tidak dapat dilakukan secara real-time; (4) sulit untuk diterapkan pada skala besar (Upadhyay et al., 2025). Di daerah pedesaan dengan akses terbatas ke layanan ekstensi pertanian, petani seringkali tidak dapat mendapatkan diagnosis yang cepat, sehingga penanganan penyakit menjadi terlambat dan kerugian menjadi lebih besar.

## 2.2 Deep Learning untuk Klasifikasi Gambar

Deep learning, khususnya Convolutional Neural Networks (CNN), telah merevolusi bidang computer vision dan klasifikasi gambar dalam dekade terakhir. CNN mampu mempelajari fitur hierarkis dari gambar secara otomatis, mulai dari fitur sederhana seperti tepi dan tekstur hingga fitur kompleks seperti pola dan objek, tanpa memerlukan ekstraksi fitur manual (Shoaib et al., 2023). Kemampuan ini membuat CNN sangat cocok untuk tugas klasifikasi gambar kompleks seperti diagnosis penyakit tanaman.

![Gambar 2.2: Evolusi deep learning dalam deteksi penyakit tanaman](images/gambar_2_2_dl_evolution.png)

**Gambar 2.2: Evolusi deep learning dalam deteksi penyakit tanaman (2015-2025)**

Sumber: Adaptasi dari Upadhyay et al. (2025)

Arsitektur CNN populer yang digunakan untuk klasifikasi gambar termasuk AlexNet, VGG, ResNet, Inception, dan EfficientNet. EfficientNetV2, yang digunakan dalam penelitian ini, merupakan arsitektur terbaru yang menyeimbangkan akurasi dan efisiensi komputasi melalui compound scaling yang mengoptimalkan kedalaman, lebar, dan resolusi secara bersamaan (Upadhyay et al., 2025). EfficientNetV2-B0, varian terkecil dari keluarga EfficientNetV2, memiliki ukuran model yang relatif kecil (~22 MB) namun tetap mencapai akurasi yang kompetitif, menjadikannya cocok untuk deployment pada berbagai platform termasuk mobile devices dan edge computing (Wang et al., 2025).

Transfer learning merupakan teknik penting dalam deep learning untuk klasifikasi gambar dengan dataset terbatas. Model yang dilatih pada dataset besar seperti ImageNet dapat digunakan sebagai feature extractor dan di-fine-tune pada dataset target yang lebih kecil, mengurangi waktu dan sumber daya yang dibutuhkan untuk pelatihan sambil tetap mencapai performa yang baik (Shafik et al., 2024). Dalam konteks diagnosis penyakit tanaman, transfer learning dari ImageNet telah terbukti efektif karena fitur visual dasar seperti tekstur dan pola yang dipelajari dari ImageNet dapat ditransfer ke domain penyakit tanaman (Bouacida et al., 2024).

## 2.3 MLOps (Machine Learning Operations)

Machine Learning Operations (MLOps) adalah disiplin yang menggabungkan praktik DevOps dengan machine learning untuk mengelola siklus hidup model machine learning secara end-to-end secara efisien dan skalabel (Behl, 2025). MLOps mencakup seluruh proses dari pengembangan model, deployment, monitoring, hingga pemeliharaan model dalam produksi. Tujuan utama MLOps adalah memastikan model machine learning dapat dideploy, dipantau, dan diperbarui secara sistematis untuk mempertahankan kinerja optimal dalam jangka panjang.

![Gambar 2.3: Siklus hidup MLOps](images/gambar_2_3_mlops_lifecycle.png)

**Gambar 2.3: Siklus hidup MLOps**

Sumber: Adaptasi dari Behl (2025)

Komponen utama MLOps meliputi:

1. **Continuous Integration dan Deployment (CI/CD)**: Otomatisasi proses build, test, dan deployment model untuk memastikan perubahan dapat diterapkan dengan cepat dan aman (Behl, 2025). Tools seperti Jenkins, GitHub Actions, atau GitLab CI dapat digunakan untuk mengotomatisasi pipeline ML.

2. **Model Registry**: Sistem untuk menyimpan, versioning, dan melacak model machine learning beserta metadata seperti hyperparameters, metrics, dan dataset yang digunakan (Behl, 2025). Model registry memungkinkan tracking model version, comparison antar model, dan rollback jika diperlukan.

3. **Monitoring dan Observability**: Monitoring kinerja model secara real-time untuk mendeteksi degradation, drift, atau anomali (Behl, 2025). Indikator yang dipantau meliputi akurasi, latensi, throughput, dan distribusi prediksi. Tools seperti Prometheus, Grafana, atau MLflow dapat digunakan untuk monitoring.

4. **Drift Detection**: Mendeteksi perubahan distribusi data input atau output yang dapat menyebabkan penurunan kinerja model (Behl, 2025). Drift dapat terjadi karena perubahan kondisi lingkungan, musim, atau perilaku pengguna.

5. **Automated Retraining**: Proses untuk melatih ulang model secara otomatis ketika kinerja model menurun atau data baru tersedia (Behl, 2025). Automated retraining memastikan model selalu up-to-date dengan kondisi terkini.

6. **Feature Store**: Sistem untuk menyimpan dan mengelola fitur yang digunakan untuk training dan inference, memastikan konsistensi antara training dan production (Behl, 2025).

Implementasi MLOps dalam konteks diagnosis penyakit tanaman memungkinkan model untuk diperbarui secara berkala dengan data baru dari lapangan, dipantau kinerjanya secara real-time, dan dikelola versinya untuk memastikan kualitas diagnosis yang konsisten.

## 2.4 Active Learning

Active learning adalah paradigma pembelajaran mesin di mana model dapat secara selektif meminta label pada data yang paling informatif dari oracle (ahli manusia), sehingga meningkatkan efisiensi proses labeling (Dong et al., 2023). Berbeda dengan supervised learning tradisional yang memerlukan seluruh dataset dilabeli sebelum pelatihan, active learning memungkinkan model untuk belajar secara iteratif dengan memilih sampel yang paling bernilai untuk dilabeli.

![Gambar 2.4: Loop active learning](images/gambar_2_4_active_learning_flow.png)

**Gambar 2.4: Loop active learning**

Sumber: Adaptasi dari Dong et al. (2023)

Strategi sampling yang umum dalam active learning meliputi:

1. **Uncertainty Sampling**: Memilih sampel dengan confidence prediksi terendah atau paling tidak pasti (Dong et al., 2023). Strategi ini mengasumsikan bahwa data yang paling tidak pasti memberikan informasi paling bernilai untuk model.

2. **Query-by-Committee**: Memilih sampel yang memiliki perbedaan prediksi terbesar antar model dalam komite (Dong et al., 2023). Sampel dengan perbedaan besar menunjukkan ketidakpastian dan potensi informasi tinggi.

3. **Expected Model Change**: Memilih sampel yang diharapkan memberikan perubahan terbesar pada model jika dilabeli (Dong et al., 2023).

4. **Diversity Sampling**: Memilih sampel yang paling beragam untuk meningkatkan cakupan ruang fitur (Dong et al., 2023).

Dalam konteks diagnosis penyakit tanaman, active learning dapat digunakan untuk mengidentifikasi prediksi dengan confidence rendah atau yang menunjukkan gejala tidak biasa, kemudian meminta validasi dari ahli pertanian. Data yang divalidasi kemudian digunakan untuk retraining model secara iteratif, meningkatkan akurasi dan robustness model terhadap kondisi lapangan yang bervariasi (Wu et al., 2023). Pendekatan ini tidak hanya meningkatkan kualitas model tetapi juga mengurangi biaya labeling dengan fokus pada data yang paling bernilai.

## 2.5 Model Registry dan Versioning

Model registry adalah sistem untuk menyimpan, versioning, dan melacak model machine learning beserta metadata terkait (Behl, 2025). Model registry menyediakan single source of truth untuk semua model yang dideploy dalam produksi, memungkinkan tracking model version, comparison antar model, dan rollback jika diperlukan.

Komponen utama model registry meliputi:

1. **Model Storage**: Menyimpan artifact model (file .pt, .pkl, dll) dan metadata terkait (Behl, 2025).

2. **Metadata Management**: Menyimpan informasi seperti hyperparameters, training metrics, dataset version, training environment, dan deployment status (Behl, 2025).

3. **Versioning**: Mengelola model version dengan sistem penomoran yang konsisten (misalnya v1, v2, v3) (Behl, 2025).

4. **Lineage Tracking**: Melacak asal-usul model termasuk dataset yang digunakan, hyperparameters, dan proses training (Behl, 2025).

5. **Deployment Tracking**: Melacak status deployment model (deployed, candidate, retired) dan environment di mana model dideploy (Behl, 2025).

Dalam penelitian ini, model registry diimplementasikan menggunakan database SQLite dengan tabel `model_registry` yang menyimpan informasi model version, accuracy, macro F1, feedback samples used, training platform, dan status deployment. Registry ini memungkinkan tracking evolusi model dari v1 (baseline), v2 (retraining tanpa feedback), hingga v3 (active learning retraining).

## 2.6 Deployment dan Containerization

Deployment model machine learning ke produksi memerlukan pendekatan yang sistematis untuk memastikan konsistensi lingkungan, skalabilitas, dan kemudahan pemeliharaan. Docker containerization telah menjadi standar industri untuk deployment aplikasi karena menyediakan lingkungan yang konsisten dan terisolasi (Behl, 2025).

Docker containerization mengemas aplikasi beserta dependensinya dalam container yang dapat berjalan di mana saja, memastikan konsistensi antara development, testing, dan production environments (Behl, 2025). Docker Compose mempermudah orchestration multi-container dengan mendefinisikan konfigurasi service dalam file YAML.

DigitalOcean Droplet menyediakan cloud infrastructure yang mudah digunakan untuk deployment aplikasi dengan biaya terjangkau (Behl, 2025). Droplet adalah virtual private server (VPS) yang dapat dikonfigurasi sesuai kebutuhan, menyediakan akses global, skalabilitas, dan kemudahan pemeliharaan.

Dalam penelitian ini, sistem dideploy pada DigitalOcean Droplet dengan Docker Compose yang menjalankan dua service: (1) API service (FastAPI) untuk inference dan (2) Web service (Streamlit) untuk user interface. Volume mount digunakan untuk persistensi data dan model, memastikan data tidak hilang saat container di-restart.

## 2.7 Penelitian Terkait

### 2.7.1 Deep Learning untuk Penyakit Tanaman

Shoaib et al. (2023) melakukan review komprehensif tentang deep learning untuk deteksi penyakit tanaman dari tahun 2015-2022, menganalisis 278 artikel. Review ini membahas teknik, model, dataset, dan tren dalam deep learning untuk penyakit tanaman, serta tantangan seperti ketersediaan data, kualitas citra, dan generalisasi model. Penelitian ini menekankan bahwa sebagian besar penelitian dilakukan pada dataset "bersih" yang tidak mencerminkan kondisi lapangan nyata.

Upadhyay et al. (2025) melakukan review yang lebih terkini dengan fokus pada deep learning dan computer vision untuk penyakit tanaman, menghubungkan riset ke production environments. Review ini mencakup 278 artikel dan membahas teknik, model, dataset, dan tren terkini, serta bagaimana menghubungkan riset ke lingkungan produksi.

Wang et al. (2025) melakukan review tentang aplikasi deep learning untuk deteksi penyakit dan hama tanaman, dengan fokus pada integrasi IoT/edge, kebutuhan model ringan dan robust untuk kondisi lapang, dan interpretability model agar dapat dipakai praktisi lapangan.

### 2.7.2 Dataset dan Tantangan In-the-Wild

Wei et al. (2024) memperkenalkan dataset PlantWild untuk in-the-wild multimodal plant disease recognition dengan citra dan deskripsi teks. Dataset ini menyoroti masalah small inter-class discrepancy dan large intra-class variance di lapang, di mana gambar dari kelas yang sama dapat terlihat sangat berbeda karena variasi kondisi. PlantWild menyediakan baseline yang realistis untuk mengembangkan sistem diagnosis yang robust terhadap kondisi lapangan.

Wei et al. (2024) juga mengembangkan sistem "Snap and Diagnose" yang merupakan sistem multimodal retrieval berbasis PlantWild dan model CLIP vision-language. Sistem ini memungkinkan user untuk upload gambar atau teks untuk retrieval penyakit, menunjukkan konsep sistem web praktis untuk diagnosis in-the-wild, namun belum mencakup MLOps lifecycle seperti monitoring dan retraining.

Xu et al. (2024) memberikan perspektif tentang dataset penyakit tanaman di era deep learning, menjelaskan mengapa dataset yang ada "menggampangkan" real-world dan bagaimana merancang dataset yang lebih mirip lapangan. Penelitian ini menekankan pentingnya dataset yang mencerminkan kondisi nyata untuk pengembangan model yang robust.

Wu et al. (2023) mengembangkan MSUN (Multi-Scale Unsupervised Network) untuk unsupervised domain adaptation dari laboratorium ke lapang, mengatasi domain shift dengan banyak data unlabeled. Penelitian ini menunjukkan pentingnya domain adaptation untuk mengatasi perbedaan antara data lab dan data lapang.

Dong et al. (2023) mengembangkan model open-world yang dapat mendeteksi penyakit tak dikenal dan melakukan pembaruan kelas secara dinamis. Model ini menggunakan unknown-aware RPN, class-agnostic ROI, dan sample replay untuk incremental learning, menunjukkan pendekatan untuk menangani unknown classes dalam deployment.

### 2.7.3 Sistem Real-time dan Aplikasi Web

Rahman et al. (2025) mengembangkan sistem real-time monitoring yang menggabungkan beberapa dataset (termasuk PlantVillage), melatih banyak CNN, dan membangun web dan mobile app untuk upload/capture citra daun, deteksi penyakit, dan memberi saran tindakan pengobatan. Sistem ini menunjukkan aplikasi akhir yang komprehensif namun belum membahas siklus hidup model (monitoring drift, retraining, model registry).

### 2.7.4 MLOps dan Deployment

Behl (2025) mengembangkan kerangka MLOps generik dengan CI/CD, monitoring otomatis, drift detection, rollback menggunakan Jenkins, Docker, Kubernetes, MLflow, dan Airflow. Penelitian ini menunjukkan indikator seperti akurasi, waktu deployment, dan deteksi drift, serta memberikan framework yang dapat diadaptasi untuk berbagai aplikasi machine learning.

Albahli (2025) mengembangkan AgriFusionNet, model ringan untuk diagnosis penyakit berbasis RGB drone, multispektral, dan data sensor lingkungan. Penelitian ini menekankan kebutuhan generalisasi di kondisi lingkungan berbeda dan edge suitability, menunjukkan pentingnya integrasi multimodal data untuk diagnosis yang robust.

### 2.7.5 Gap Penelitian

**Tabel 2.1: Ringkasan penelitian terkait**

| Paper | Fokus | Kontribusi | Relevansi |
|---|---|---|---|
| Shoaib et al. (2023) | Review DL penyakit tanaman | Merangkum tren CNN/deep learning | Dasar teoritis DL |
| Upadhyay et al. (2025) | Review CV/DL precision agriculture | Menjelaskan model, dataset, dan tren production-oriented | Mendukung pemilihan model ringan dan deployment |
| Wei et al. (2024) PlantWild | Benchmark in-the-wild | Dataset realistis kondisi lapangan | Dasar dataset dan tantangan domain nyata |
| Wei et al. (2024) Snap and Diagnose | Sistem retrieval multimodal | Aplikasi web diagnosis/retrieval penyakit | Pembanding sistem end-user |
| Behl (2025) | MLOps framework | CI/CD, monitoring, drift, rollback | Dasar lifecycle operasional |
| Dong et al. (2023) | Open-world plant disease detection | Deteksi unknown dan pembaruan kelas | Batasan dan future work |
| Wu et al. (2023) | Domain adaptation | Adaptasi lab-to-field | Batasan domain shift |
| Rahman et al. (2025) | Real-time monitoring system | Web/mobile diagnosis penyakit daun | Pembanding aplikasi real-time |

**Tabel 2.2: Gap penelitian**

| Area | Studi Terdahulu | Gap | Posisi AgriMLOps |
|---|---|---|---|
| Model deep learning | Fokus pada akurasi klasifikasi | Operasionalisasi model belum lengkap | Inference API, monitoring, registry |
| Dataset in-the-wild | PlantWild membahas benchmark | Belum menjadi platform MLOps produksi | PlantWild subset untuk sistem web/API |
| Aplikasi web | Snap and Diagnose dan Rahman et al. membangun aplikasi | Feedback, registry, dan retraining belum kuat | Feedback logging dan active learning queue |
| MLOps | Behl membahas framework generik | Tidak spesifik penyakit tanaman | Implementasi domain pertanian |
| Active learning/open-world | Dong membahas unknown class | AgriMLOps belum true open-world | Controlled active learning simulation sebagai tahap awal |
| Domain adaptation | Wu membahas lab-to-field adaptation | AgriMLOps belum menerapkan UDA eksplisit | Dicatat sebagai limitasi dan future work |

Berdasarkan tinjauan pustaka di atas, tidak ada paper yang secara eksplisit membahas **platform web MLOps lengkap khusus diagnosis penyakit tanaman in-the-wild** dengan kombinasi deep learning, active learning, dan model monitoring. Penelitian-penelitian yang ada fokus pada:

1. Model development tanpa MLOps lifecycle (sebagian besar penelitian)
2. Sistem web/mobile tanpa MLOps (Rahman et al., 2025; Wei et al., 2024)
3. MLOps generik tanpa konteks pertanian (Behl, 2025)
4. Dataset dan domain adaptation tanpa platform lengkap (Wei et al., 2024; Wu et al., 2023; Dong et al., 2023)

Penelitian ini mengisi gap dengan mengintegrasikan komponen yang sudah ada ke dalam satu platform MLOps khusus penyakit tanaman lapang, mencakup model inference, logging, feedback collection, active learning, monitoring, model registry, dan deployment cloud.
