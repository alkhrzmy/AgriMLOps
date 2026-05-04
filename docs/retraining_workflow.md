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

## Optional Controlled Feedback Simulation untuk v3

Controlled feedback simulation digunakan untuk menguji retraining loop dengan dataset eksternal atau holdout berlabel sebelum feedback petani asli tersedia.

Simulasi ini bukan feedback petani nyata dan tidak boleh diklaim sebagai data petani asli.

Jalankan simulasi dari dataset berfolder label:

```bash
python scripts/simulate_validated_feedback.py \
  --dataset-root /path/to/labeled_dataset \
  --api-base-url http://159.65.139.148:8000 \
  --max-images 50
```

Script hanya memproses folder label yang cocok dengan `models/label_map.json`. Prediksi salah atau low-confidence akan dikirim sebagai feedback lalu divalidasi otomatis sebagai controlled validated feedback.

## Workflow v3 dengan Controlled Validated Feedback

1. Jalankan `scripts/simulate_validated_feedback.py` terhadap production API.

2. SSH ke droplet.

3. Jalankan export validated feedback:

```bash
python scripts/export_validated_feedback.py
```

4. Download ZIP terbaru dari `data/retraining/`.

5. Upload ZIP ke Kaggle sebagai feedback dataset.

6. Train di Kaggle dengan konfigurasi:

```python
MODEL_VERSION = "v3"
USE_VALIDATED_FEEDBACK = True
FEEDBACK_ZIP_PATH = "/kaggle/input/agrimlops-feedback/validated_feedback.zip"
EXISTING_LABEL_MAP_PATH = "/kaggle/input/agrimlops-v2/label_map.json"
BASE_MODEL_PATH = "/kaggle/input/agrimlops-v2/model_v2.pt"
```

7. Pastikan metadata v3 mencatat:

```text
base_model_version: v2
feedback_samples_used: > 0
notes: retrained using controlled validated feedback simulation + PlantWild subset
```

8. Compare v2 vs v3 sebelum promote.

## Prinsip Data dan Evaluasi

- Feedback user adalah weak label.
- Ground truth baru hanya berasal dari validasi admin atau penyuluh.
- Controlled feedback simulation berasal dari dataset eksternal/holdout berlabel, bukan feedback petani nyata.
- Hanya item `active_learning_queue` dengan `status=validated` dan `validated_label` valid yang boleh masuk retraining.
- Feedback tervalidasi hanya ditambahkan ke `train_df`.
- Feedback tidak dimasukkan ke validation/test agar evaluasi tetap bersih.
- `label_map.json` harus dipakai ulang supaya output model tetap kompatibel dengan API dan UI.

## Kenapa Tetap Termasuk MLOps Lifecycle

Workflow ini tetap MLOps karena sistem mengelola:

- data baru dari production;
- validasi data sebelum training;
- model candidate v2;
- evaluasi dan comparison v1 vs v2;
- promotion model secara aman melalui `CURRENT_MODEL_VERSION`;
- redeployment service dengan artifact versi baru.
