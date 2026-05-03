# Devin AI Handoff Summary - AgriMLOps PlantWild

## Project Goal

AgriMLOps PlantWild is an INNOVEST MVP web app for in-the-wild plant disease diagnosis. The product direction is not just image classification, but an MLOps lifecycle prototype: model inference, prediction logging, user feedback, active learning queue, monitoring, and retraining workflow.

## Current Stack

- Backend: FastAPI
- Frontend: Streamlit
- Database target: SQLite via SQLAlchemy
- Model training: PyTorch + timm
- Main model: `tf_efficientnetv2_b0`
- Dataset: Hugging Face `uqtwei2/PlantWild`
- Deployment target: DigitalOcean Droplet with Docker Compose

## Training Workflow Decision

Heavy training is not done locally. Training is done on Kaggle GPU using:

```text
notebooks/kaggle_train_plantwild.py
```

The script downloads PlantWild from Hugging Face, extracts `plantwild.zip`, scans images recursively, selects top 15 classes, performs stratified train/val/test split, trains EfficientNetV2-B0, evaluates test metrics, and exports artifacts to:

```text
/kaggle/working/agrimlops_artifacts.zip
```

## Imported Model Artifact

Kaggle artifact has been imported into the local repo.

Required artifact verification passed with:

```bash
python scripts/verify_artifacts.py
```

Current model metadata:

```json
{
  "model_version": "v1",
  "model_name": "tf_efficientnetv2_b0",
  "dataset": "uqtwei2/PlantWild subset",
  "num_classes": 15,
  "input_size": 224,
  "accuracy": 0.8217237308146399,
  "macro_f1": 0.8085807733841014,
  "created_at": "2026-05-03T16:49:29.915013+00:00",
  "notes": "trained on Kaggle GPU"
}
```

Model file size:

```text
models/model_v1.pt: ~22.8 MB
```

This is below GitHub's 100 MB hard limit and can be committed directly.

## Implemented So Far

- Project scaffold for FastAPI, Streamlit, SQLite/MLOps loop modules, Docker, docs, and scripts.
- Kaggle training script with archive extraction support for PlantWild.
- Artifact contract in `artifacts_contract/ARTIFACTS.md`.
- Artifact verification script in `scripts/verify_artifacts.py`.
- `src/predict.py` can load:
  - `models/model_v1.pt`
  - `models/label_map.json`
  - `models/model_v1_metadata.json`
- `src/predict.py` returns predicted label, confidence, top-3 predictions, recommendation, and `needs_review` flag.
- FastAPI `/health` reports model artifact status.
- FastAPI `/model/current` reports current model info.
- Streamlit Model Registry reads and displays model metadata.
- Dockerfiles and Docker Compose exist for API and web services.

## Important Limitation

The project is artifact-ready, but the full user-facing diagnosis flow is not complete yet.

Missing or incomplete MVP features:

- FastAPI `POST /predict` upload endpoint.
- FastAPI `POST /feedback` endpoint.
- SQLite schema and insert/query functions for prediction logs, feedback logs, active learning queue, and model registry.
- Streamlit Diagnosis page calling the API `/predict` endpoint.
- Streamlit feedback form connected to backend.
- Monitoring dashboard backed by SQLite data.
- Active learning validation UI connected to backend.

## Recommended Next Decisions

1. Implement SQLite MLOps loop in `src/database.py`.
2. Implement FastAPI endpoints:
   - `POST /predict`
   - `POST /feedback`
   - `GET /monitoring/summary`
   - `GET /active-learning/queue`
   - `POST /active-learning/{id}/validate`
3. Connect Streamlit Diagnosis page to API prediction endpoint.
4. Connect feedback form to API.
5. Smoke test Docker Compose locally or on the droplet.
6. Add basic admin/auth protection before production exposure.

## Deployment Readiness

DigitalOcean Docker Compose can run the current services if Docker is installed and artifacts are present. However, current deployment is best described as an artifact/model-registry smoke test, not the complete diagnosis product, until `/predict` and database feedback endpoints are implemented.
