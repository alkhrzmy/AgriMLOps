# BAB II
# KAJIAN PUSTAKA

## 2.1 Penyakit Tanaman dan Diagnosis Berbasis Citra

Penyakit tanaman menyebabkan penurunan kualitas dan kuantitas hasil pertanian. Shoaib et al. (2023) menjelaskan bahwa kerugian akibat penyakit tanaman dapat mencapai skala ekonomi yang besar, sedangkan Rahman et al. (2025) menekankan pentingnya deteksi dini untuk menjaga produktivitas. Diagnosis manual masih digunakan, tetapi cara tersebut bergantung pada keahlian pengamat dan sulit dilakukan secara cepat pada skala luas.

![Gambar 2.1: Dampak penyakit tanaman pada hasil panen global](images/gambar_2_1_plant_disease_impact.png)

**Gambar 2.1: Dampak penyakit tanaman pada hasil panen global**

Sumber: Adaptasi dari Shoaib et al. (2023)

Deep learning telah banyak digunakan untuk mengenali penyakit tanaman dari citra daun. CNN dapat mempelajari pola warna, tekstur, dan bentuk lesi secara otomatis sehingga sesuai untuk klasifikasi gambar (Shoaib et al., 2023). Transfer learning juga mempercepat pelatihan karena model memanfaatkan representasi visual yang telah dipelajari pada dataset besar (Shafik et al., 2024). EfficientNetV2-B0 dipilih dalam penelitian ini karena ukuran model relatif kecil dan tetap kompetitif untuk klasifikasi citra.

![Gambar 2.2: Evolusi deep learning dalam deteksi penyakit tanaman](images/gambar_2_2_dl_evolution.png)

**Gambar 2.2: Evolusi deep learning dalam deteksi penyakit tanaman (2015-2025)**

Sumber: Adaptasi dari Upadhyay et al. (2025)

## 2.2 Dataset In-the-Wild dan Domain Shift

Banyak studi deteksi penyakit tanaman menggunakan dataset dengan kondisi visual yang relatif bersih. Namun, citra lapangan memiliki latar belakang kompleks, pencahayaan beragam, dan kualitas kamera yang tidak seragam. Wei et al. (2024) memperkenalkan PlantWild untuk menekankan dua tantangan utama: small inter-class discrepancy, yaitu kelas berbeda tampak mirip, dan large intra-class variance, yaitu kelas yang sama dapat tampak sangat berbeda. Tantangan ini membuat evaluasi pada dataset in-the-wild lebih realistis dibandingkan dataset laboratorium.

Wu et al. (2023) menunjukkan bahwa model yang dilatih pada data laboratorium dapat mengalami penurunan performa ketika diterapkan pada data lapangan karena domain shift. Penelitian ini belum menerapkan domain adaptation eksplisit, tetapi memilih PlantWild sebagai basis data agar proses pengembangan lebih dekat dengan kondisi nyata.

## 2.3 MLOps dan Active Learning

MLOps adalah pendekatan untuk mengelola siklus hidup model mulai dari pelatihan, deployment, monitoring, versioning, hingga pembaruan model. Behl (2025) menekankan bahwa MLOps membantu menjembatani eksperimen model dengan sistem produksi melalui otomasi, monitoring, dan tata kelola model. Dalam AgriMLOps, prinsip tersebut diterjemahkan menjadi API inference, prediction logging, feedback logging, model registry, dan monitoring dashboard.

![Gambar 2.3: Siklus hidup MLOps](images/gambar_2_3_mlops_lifecycle.png)

**Gambar 2.3: Siklus hidup MLOps**

Sumber: Adaptasi dari Behl (2025)

Active learning memungkinkan sistem memilih sampel yang paling informatif untuk diberi label, misalnya prediksi dengan confidence rendah. Dong et al. (2023) membahas kebutuhan pembelajaran dinamis pada deteksi penyakit tanaman ketika model menghadapi kelas atau kondisi baru. Penelitian ini menggunakan prinsip tersebut dalam bentuk active learning queue. Prediksi yang tidak pasti disimpan, divalidasi, kemudian dapat digunakan sebagai data retraining.

![Gambar 2.4: Loop active learning](images/gambar_2_4_active_learning_flow.png)

**Gambar 2.4: Loop active learning**

Sumber: Adaptasi dari Dong et al. (2023)

## 2.4 Penelitian Terkait dan Gap Penelitian

Rahman et al. (2025) membangun sistem web dan mobile untuk deteksi penyakit daun menggunakan beberapa model CNN. Wei et al. (2024) mengembangkan PlantWild serta sistem Snap and Diagnose untuk diagnosis multimodal in-the-wild. Behl (2025) membahas MLOps secara generik dengan CI/CD, monitoring, rollback, dan deployment cloud-native. Dong et al. (2023) serta Wu et al. (2023) masing-masing membahas open-world learning dan domain adaptation.

**Tabel 2.1: Ringkasan penelitian terkait**

| Paper | Fokus | Relevansi terhadap AgriMLOps |
|---|---|---|
| Shoaib et al. (2023) | Review deep learning penyakit tanaman | Dasar kebutuhan diagnosis berbasis citra |
| Wei et al. (2024) PlantWild | Benchmark in-the-wild | Dasar pemilihan dataset |
| Rahman et al. (2025) | Web/mobile diagnosis daun | Pembanding aplikasi end-user |
| Behl (2025) | Framework MLOps | Dasar lifecycle operasional |
| Dong et al. (2023) | Open-world dan dynamic learning | Inspirasi active learning, bukan klaim open-world |
| Wu et al. (2023) | Domain adaptation | Batasan terkait domain shift |

**Tabel 2.2: Gap penelitian**

| Area | Gap | Posisi AgriMLOps |
|---|---|---|
| Model klasifikasi | Banyak studi berhenti pada akurasi model | Menambahkan logging, monitoring, dan registry |
| Aplikasi web | Sistem diagnosis belum selalu memiliki retraining loop | Menyediakan feedback dan active learning queue |
| MLOps | Framework generik belum spesifik pertanian | Menerapkan MLOps pada diagnosis penyakit tanaman |
| Data lapangan | Domain shift masih menjadi tantangan | Menggunakan PlantWild subset in-the-wild |

Berdasarkan tinjauan tersebut, kontribusi AgriMLOps terletak pada integrasi model diagnosis, MLOps lifecycle, dan active learning queue dalam satu prototipe yang dapat diuji secara end-to-end. Penelitian ini tidak mengklaim menyelesaikan open-world detection atau domain adaptation, tetapi menyiapkan fondasi operasional untuk pengembangan ke arah tersebut.
