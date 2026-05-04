# Controlled Feedback Simulation

Controlled feedback simulation adalah cara terkontrol untuk menguji loop active learning dan retraining sebelum feedback petani asli tersedia dalam jumlah cukup.

## Tujuan

- Menjalankan batch inference ke production API dari dataset berlabel.
- Mengirim feedback otomatis berdasarkan ground truth dari struktur folder dataset.
- Memvalidasi item active learning secara otomatis jika prediksi salah atau benar tetapi confidence rendah.
- Menghasilkan validated feedback yang bisa diexport untuk retraining v3 di Kaggle GPU.

## Format Dataset

Dataset harus generic berbasis folder label:

```text
dataset_root/
  class_name_1/
    image1.jpg
    image2.jpg
  class_name_2/
    image3.jpg
```

Nama folder harus sama dengan label di `models/label_map.json`.

## Menjalankan Simulasi

```bash
python scripts/simulate_validated_feedback.py \
  --dataset-root /path/to/labeled_dataset \
  --api-base-url http://159.65.139.148:8000 \
  --max-images 50 \
  --confidence-threshold 0.70
```

Dry run:

```bash
python scripts/simulate_validated_feedback.py \
  --dataset-root /path/to/labeled_dataset \
  --dry-run
```

Output report:

```text
reports/controlled_feedback_simulation.json
```

## Aturan Label

- Script hanya memproses folder yang namanya cocok dengan label di `models/label_map.json`.
- Folder dengan nama label di luar label map akan dilewati.
- Hal ini menjaga agar retraining tetap kompatibel dengan output model production.

## Perilaku Feedback

- Jika prediksi benar dan confidence tinggi, script mengirim feedback `correct`.
- Jika prediksi salah, script mengirim feedback `incorrect` dengan `suggested_label` sebagai ground truth lalu memvalidasi queue item.
- Jika prediksi benar tetapi confidence rendah, script mengirim feedback `unsure` dengan `suggested_label` sebagai ground truth lalu memvalidasi queue item.

## Batasan dan Etika Data

- Simulasi ini menggunakan dataset eksternal atau holdout berlabel.
- Simulasi ini bukan feedback petani asli.
- Jangan mengklaim hasil simulasi sebagai data petani nyata.
- Feedback user asli tetap harus divalidasi admin atau penyuluh sebelum dipakai untuk retraining.

## Hubungan dengan Retraining v3

Setelah simulasi selesai, export validated feedback dari droplet:

```bash
python scripts/export_validated_feedback.py
```

Upload ZIP hasil export ke Kaggle, lalu train `MODEL_VERSION = "v3"` dengan `model_v2.pt` sebagai base model.
