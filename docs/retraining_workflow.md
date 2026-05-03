# Semi-Manual Retraining Workflow

Retraining AgriMLOps PlantWild dilakukan secara semi-manual karena training GPU tetap berjalan di Kaggle, bukan di droplet atau laptop lokal.

## Workflow

1. Jalankan app dan kumpulkan prediction logs.

2. User memberi feedback melalui halaman Diagnosis Tanaman.

3. Admin atau penyuluh memvalidasi item di Active Learning Queue.

4. Export feedback tervalidasi:

```bash
python scripts/export_validated_feedback.py
```

5. Upload ZIP feedback ke Kaggle sebagai dataset input.

6. Upload artifact v1 ke Kaggle sebagai dataset input:

```text
model_v1.pt
label_map.json
```

7. Jalankan Kaggle notebook/script dengan konfigurasi:

```python
MODEL_VERSION = "v2"
USE_VALIDATED_FEEDBACK = True
FEEDBACK_ZIP_PATH = "/kaggle/input/agrimlops-feedback/validated_feedback.zip"
EXISTING_LABEL_MAP_PATH = "/kaggle/input/agrimlops-v1/label_map.json"
BASE_MODEL_PATH = "/kaggle/input/agrimlops-v1/model_v1.pt"
```

8. Download output Kaggle:

```text
/kaggle/working/agrimlops_artifacts_v2.zip
```

9. Extract artifact v2 ke repo lokal.

10. Verify artifact v2:

```bash
python scripts/verify_artifacts.py --version v2
```

11. Compare model v1 dan v2:

```bash
python scripts/compare_models.py
```

12. Jika v2 lebih baik, promote candidate:

```bash
python scripts/promote_model.py --version v2
```

13. Redeploy Docker:

```bash
docker compose up -d --build
```

## Artifact v2 yang Diharapkan

```text
models/model_v2.pt
models/model_v2_metadata.json
models/label_map.json
reports/metrics_v2.json
reports/confusion_matrix_v2.png
reports/classification_report_v2.csv
reports/retraining_dataset_summary_v2.json
```

## Prinsip Data dan Evaluasi

- Feedback user adalah weak label.
- Ground truth baru hanya berasal dari validasi admin atau penyuluh.
- Hanya item `active_learning_queue` dengan `status=validated` dan `validated_label` valid yang boleh masuk retraining.
- Feedback tervalidasi hanya ditambahkan ke `train_df`.
- Feedback tidak dimasukkan ke validation/test agar evaluasi tetap bersih.
- `label_map.json` v1 harus dipakai ulang supaya output model tetap kompatibel dengan API dan UI.

## Kenapa Tetap Termasuk MLOps Lifecycle

Workflow ini tetap MLOps karena sistem mengelola:

- data baru dari production;
- validasi data sebelum training;
- model candidate v2;
- evaluasi dan comparison v1 vs v2;
- promotion model secara aman melalui `CURRENT_MODEL_VERSION`;
- redeployment service dengan artifact versi baru.
