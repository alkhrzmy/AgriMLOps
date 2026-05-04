# BAB III
# RANCANGAN DAN METODE IMPLEMENTASI

## 3.1 Pendekatan Penelitian

Penelitian ini menggunakan pendekatan rekayasa sistem berbasis MLOps. Objek yang dikembangkan adalah prototipe AgriMLOps untuk diagnosis penyakit tanaman in-the-wild. Alur penelitian dimulai dari pemilihan dataset PlantWild subset, pelatihan model EfficientNetV2-B0, integrasi API dan antarmuka web, pencatatan prediksi, pengumpulan feedback, validasi active learning queue, retraining model, dan evaluasi antar versi model.

Pengembangan model dilakukan secara bertahap. Model v1 digunakan sebagai baseline dengan bobot awal ImageNet. Model v2 digunakan untuk menguji pipeline retraining tanpa tambahan feedback tervalidasi. Model v3 dilatih menggunakan transfer learning dari v2 dan menambahkan 9 sampel controlled validated feedback. Dengan urutan ini, penelitian dapat membedakan peningkatan karena retraining biasa dan peningkatan yang melibatkan feedback tervalidasi.

Siklus perbaikan sistem berlangsung melalui inference, logging, feedback, validasi, retraining, dan monitoring. Pengguna mengunggah citra melalui Streamlit, FastAPI menjalankan prediksi, dan SQLite mencatat hasil prediksi. Apabila confidence rendah atau pengguna memberi feedback tidak yakin/salah, item masuk ke active learning queue. Data yang sudah divalidasi diekspor untuk retraining sehingga model baru dapat dibandingkan dengan versi sebelumnya.

## 3.2 Rancangan Sistem

AgriMLOps terdiri dari tiga komponen utama. Komponen pertama adalah backend FastAPI yang menyediakan endpoint prediksi, feedback, monitoring, active learning, dan registry model. Komponen kedua adalah frontend Streamlit yang menjadi antarmuka diagnosis dan pemantauan. Komponen ketiga adalah SQLite yang menyimpan prediction logs, feedback logs, active learning queue, dan model registry. Rancangan ini dipilih agar prototipe tetap ringan, mudah direplikasi, dan cukup untuk membuktikan alur MLOps pada skala MVP.

![Gambar 3.1: Arsitektur sistem AgriMLOps](images/gambar_3_1_system_architecture.png)

**Gambar 3.1: Arsitektur sistem AgriMLOps**

Sumber: Hasil implementasi penelitian

Model klasifikasi menggunakan EfficientNetV2-B0 dengan ukuran input 224 x 224 piksel dan 15 kelas penyakit tanaman. Model v1 dan v2 memakai bobot awal ImageNet, sedangkan v3 menggunakan model v2 sebagai base model. Konfigurasi utama pelatihan menggunakan batch size 32, learning rate 0,0003, optimizer AdamW, scheduler CosineAnnealingLR, dan augmentasi citra berupa RandomResizedCrop, RandomHorizontalFlip, ColorJitter, serta normalisasi ImageNet.

Database dirancang untuk mencatat jejak model dan interaksi pengguna. Tabel prediction_logs menyimpan identitas prediksi, path gambar, label prediksi, confidence, top-k predictions, status needs_review, dan timestamp. Tabel feedback_logs menyimpan feedback pengguna atau evaluator. Tabel active_learning_queue menyimpan item yang perlu divalidasi, termasuk reason, status, dan validated_label. Tabel model_registry menyimpan metadata model, metrik, jumlah feedback yang digunakan, dan status deployment.

![Gambar 3.2: Skema database](images/gambar_3_2_database_schema.png)

**Gambar 3.2: Skema database AgriMLOps**

Sumber: Hasil implementasi penelitian

API FastAPI menyediakan delapan endpoint utama, yaitu /health, /model/current, /model/registry, /predict, /feedback, /monitoring/summary, /active-learning/queue, dan /active-learning/{item_id}/validate. Endpoint tersebut mewakili kebutuhan dasar sistem: mengecek kesehatan layanan, membaca metadata model, menjalankan prediksi, menyimpan feedback, memantau ringkasan sistem, dan memvalidasi item active learning.

![Gambar 3.3: Struktur API endpoints](images/gambar_3_3_api_endpoints.png)

**Gambar 3.3: Struktur API endpoints FastAPI**

Sumber: Hasil implementasi penelitian

Antarmuka Streamlit menyediakan halaman Diagnosis, Feedback, Monitoring Dashboard, Active Learning Queue, dan Model Registry. Halaman Diagnosis dipakai untuk mengunggah citra dan membaca hasil prediksi. Halaman Feedback dan Active Learning Queue dipakai untuk mendukung validasi data. Halaman Monitoring Dashboard dan Model Registry dipakai untuk memantau prediksi, feedback, serta perbandingan versi model.

![Gambar 3.4: Struktur halaman Streamlit](images/gambar_3_4_streamlit_pages.png)

**Gambar 3.4: Struktur halaman Streamlit UI**

Sumber: Hasil implementasi penelitian

Deployment menggunakan Docker Compose dengan dua layanan, yaitu API service dan Web service. API service memuat model PyTorch, label map, database, serta endpoint FastAPI. Web service menjalankan Streamlit dan berkomunikasi dengan API melalui HTTP. Volume ./data, ./models, dan ./reports digunakan agar database, file upload, artifact model, dan laporan metrik tetap persisten.

![Gambar 3.5: Arsitektur deployment Docker Compose](images/gambar_3_5_deployment_architecture.png)

**Gambar 3.5: Arsitektur deployment Docker Compose**

Sumber: Hasil implementasi penelitian

## 3.3 Data, Alat, dan Konfigurasi

Dataset utama adalah PlantWild dari Hugging Face. Penelitian menggunakan 15 kelas teratas dengan total 5.645 gambar. Data dibagi secara stratified menjadi 70% data latih, 15% data validasi, dan 15% data uji. Pada model v3, 9 sampel controlled validated feedback ditambahkan ke data latih. Sampel tersebut berasal dari simulasi berbasis data berlabel, bukan dari feedback petani asli.

**Tabel 3.1: Spesifikasi teknologi**

| Komponen | Teknologi | Fungsi |
|---|---|---|
| Backend | FastAPI | Inference, feedback, monitoring API |
| Frontend | Streamlit | UI diagnosis dan dashboard |
| Model | EfficientNetV2-B0 | Klasifikasi penyakit tanaman |
| Database | SQLite | Logs, feedback, queue, registry |
| Training | Kaggle GPU | Training dan retraining model |
| Deployment | Docker Compose dan DigitalOcean | Uji akses web/API |

**Tabel 3.2: Dataset PlantWild subset**

| Aspek | Nilai |
|---|---|
| Sumber | Hugging Face uqtwei2/PlantWild |
| Jumlah kelas | 15 kelas top |
| Total gambar | 5.645 |
| Split | 70/15/15 |
| Train v1/v2 | 3.952 |
| Train v3 | 3.961, termasuk 9 controlled validated feedback samples |
| Val/Test | 847/847 |
| Input size | 224 x 224 |

## 3.4 Tahapan Implementasi

Tahap pertama adalah pelatihan baseline model v1 pada Kaggle GPU. PlantWild diunduh, dipindai, difilter menjadi 15 kelas, lalu dibagi menjadi train, validation, dan test set. EfficientNetV2-B0 dilatih selama 8 epoch dan menghasilkan artifact model, metadata, metrics, classification report, serta confusion matrix.

Tahap kedua adalah integrasi MLOps. Pada tahap ini, FastAPI, Streamlit, SQLite, dan Docker Compose dihubungkan menjadi sistem end-to-end. API menjalankan inference dan operasi database, sedangkan Streamlit menampilkan halaman diagnosis, feedback, monitoring, active learning, dan registry model.

Tahap ketiga adalah controlled active learning simulation. Script simulasi mengirim 75 gambar berlabel ke production API, membandingkan prediksi dengan ground truth folder, mengirim feedback, dan memvalidasi item queue yang relevan. Simulasi menghasilkan 9 item tervalidasi yang diekspor sebagai ZIP untuk retraining. Simulasi ini hanya digunakan untuk membuktikan pipeline feedback, bukan sebagai klaim feedback petani asli.

![Gambar 3.7: Workflow controlled validated feedback simulation](images/gambar_3_7_controlled_feedback_simulation.png)

**Gambar 3.7: Workflow controlled validated feedback simulation**

Sumber: Hasil implementasi penelitian

Tahap keempat adalah pelatihan model v2. Model ini dilatih ulang dengan konfigurasi yang sama seperti v1, tetapi belum menggunakan feedback tervalidasi. Tahap ini penting untuk menunjukkan bahwa retraining tanpa feedback baru tidak otomatis memberikan peningkatan besar.

Tahap kelima adalah pelatihan model v3. Model v3 menggunakan model_v2.pt sebagai base model, menambahkan 9 sampel feedback tervalidasi ke data latih, dan dilatih selama 12 epoch. Artifact v3 kemudian dievaluasi, dibandingkan, dan didaftarkan pada model registry.

![Gambar 3.6: Workflow training Kaggle](images/gambar_3_6_training_workflow.png)

**Gambar 3.6: Workflow training model di Kaggle**

Sumber: Hasil implementasi penelitian

**Tabel 3.3: Konfigurasi training v1-v2-v3**

| Versi | Base model | Epoch | Batch | LR | Feedback | Platform |
|---|---|---:|---:|---:|---:|---|
| v1 | ImageNet | 8 | 32 | 0.0003 | 0 | Kaggle GPU |
| v2 | ImageNet | 8 | 32 | 0.0003 | 0 | Kaggle GPU |
| v3 | model_v2.pt | 12 | 32 | 0.0003 | 9 | Kaggle GPU |

## 3.5 Evaluasi dan Kelayakan

Evaluasi dilakukan dengan membandingkan accuracy, macro F1, best validation macro F1, dan jumlah feedback samples used pada model v1, v2, dan v3. Selain metrik model, sistem juga dievaluasi dari keberhasilan alur MLOps, yaitu kemampuan mencatat prediksi, menerima feedback, menyimpan queue, memvalidasi label, menampilkan monitoring, dan menampilkan registry model.

Kelayakan teknis dinilai dari ukuran model, waktu inference, dan kemampuan sistem berjalan sebagai layanan web/API. Model v3 berukuran sekitar 22,8 MB sehingga masih ringan untuk prototipe. Deployment DigitalOcean dengan Docker Compose menunjukkan bahwa sistem dapat diakses melalui web dan API. Namun, deployment ini diposisikan sebagai bukti kelayakan implementasi, bukan sebagai fokus utama penelitian.

Kelayakan manfaat dilihat dari potensi sistem sebagai alat bantu diagnosis awal. Jika dikembangkan dengan data lapangan dan validasi ahli yang lebih besar, sistem dapat membantu petani atau penyuluh memperoleh diagnosis lebih cepat. Dampak tersebut tetap harus dibuktikan pada penelitian lanjutan karena penelitian ini belum melakukan uji lapangan dengan pengguna nyata.
