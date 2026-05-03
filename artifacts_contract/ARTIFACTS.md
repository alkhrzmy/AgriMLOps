# AgriMLOps Artifact Contract

Training model dilakukan di Kaggle GPU. Setelah training selesai, artifact berikut wajib diunduh dari Kaggle dan dimasukkan ke root repo lokal dengan struktur yang sama.

## Artifact Wajib

```text
models/model_v1.pt
models/label_map.json
models/model_v1_metadata.json
reports/metrics_v1.json
reports/confusion_matrix_v1.png
reports/classification_report_v1.csv
reports/class_distribution.png
reports/sample_predictions.png
```

## Metadata Model

File `models/model_v1_metadata.json` minimal berisi struktur berikut:

```json
{
  "model_version": "v1",
  "model_name": "tf_efficientnetv2_b0",
  "dataset": "uqtwei2/PlantWild subset",
  "num_classes": 15,
  "input_size": 224,
  "accuracy": 0.0,
  "macro_f1": 0.0,
  "created_at": "",
  "notes": "trained on Kaggle GPU"
}
```

## Catatan GitHub

Jika `models/model_v1.pt` lebih dari 100MB, jangan commit file tersebut ke GitHub. Simpan model di GitHub Release, Hugging Face, atau object storage, lalu simpan URL sebagai `MODEL_URL` di `.env` untuk proses deployment.

## Validasi Artifact

Setelah extract artifact ke root repo, jalankan:

```bash
python scripts/verify_artifacts.py
```
