import json
import os
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.database import (
    add_to_active_learning_queue,
    database_available,
    get_active_learning_queue,
    get_monitoring_summary,
    get_prediction,
    init_db,
    log_feedback,
    log_prediction,
    register_current_model_from_metadata,
    validate_active_learning_item,
)
from src.predict import METADATA_PATH, get_model_status, predict_image

app = FastAPI(
    title="AgriMLOps PlantWild API",
    description="Backend inference API for plant disease diagnosis and MLOps feedback loop.",
    version="0.1.0",
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/feedback/uploads"))
LOW_CONFIDENCE_THRESHOLD = 0.70


class FeedbackRequest(BaseModel):
    prediction_id: str
    user_feedback: Literal["correct", "incorrect", "unsure"]
    suggested_label: str | None = None
    comment: str | None = None


class ActiveLearningValidationRequest(BaseModel):
    validated_label: str | None = None
    validator_note: str | None = None
    status: Literal["validated", "rejected"] = "validated"


@app.on_event("startup")
def startup() -> None:
    init_db()
    register_current_model_from_metadata()


@app.get("/health")
def health_check() -> dict:
    model_status = get_model_status()
    return {
        "status": "ok",
        "service": "agrimlops-plantwild-api",
        "model_loaded": model_status["model_loaded"],
        "model_artifact_available": model_status["model_loaded"],
        "current_model_version": model_status["current_model_version"],
        "model_version": model_status["model_version"],
        "database_available": database_available(),
    }


@app.get("/model/current")
def current_model() -> dict:
    model_status = get_model_status()
    metadata = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    return {**model_status, **metadata}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    prediction_id = str(uuid.uuid4())
    upload_date = datetime.utcnow().strftime("%Y%m%d")
    target_dir = UPLOAD_DIR / upload_date
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path = target_dir / f"{prediction_id}.jpg"

    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.save(image_path, format="JPEG", quality=92)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        result = predict_image(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    model_version = get_model_status().get("model_version")
    top3_predictions = result["top_k"]
    logged = log_prediction(
        prediction_id=prediction_id,
        image_path=str(image_path),
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        top3_predictions=top3_predictions,
        recommendation=result["recommendation"],
        needs_review=result["needs_review"],
        model_version=model_version,
    )

    if result["needs_review"] or result["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        add_to_active_learning_queue(
            prediction_id=prediction_id,
            image_path=str(image_path),
            predicted_label=result["predicted_label"],
            confidence=result["confidence"],
            reason="low_confidence",
        )

    return {
        "prediction_id": logged["id"],
        "predicted_label": result["predicted_label"],
        "confidence": result["confidence"],
        "top3_predictions": top3_predictions,
        "recommendation": result["recommendation"],
        "needs_review": result["needs_review"],
        "model_version": model_version,
    }


@app.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict:
    prediction = get_prediction(payload.prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="prediction_id not found")

    feedback_row = log_feedback(
        prediction_id=payload.prediction_id,
        user_feedback=payload.user_feedback,
        suggested_label=payload.suggested_label,
        comment=payload.comment,
    )

    queue_item = None
    if payload.user_feedback in {"incorrect", "unsure"}:
        reason = "user_incorrect" if payload.user_feedback == "incorrect" else "user_unsure"
        queue_item = add_to_active_learning_queue(
            prediction_id=payload.prediction_id,
            image_path=prediction["image_path"],
            predicted_label=prediction["predicted_label"],
            confidence=prediction["confidence"],
            reason=reason,
        )

    return {"status": "ok", "feedback": feedback_row, "active_learning_item": queue_item}


@app.get("/monitoring/summary")
def monitoring_summary() -> dict:
    return get_monitoring_summary()


@app.get("/active-learning/queue")
def active_learning_queue(status: str = "pending") -> dict:
    return {"items": get_active_learning_queue(status=status)}


@app.post("/active-learning/{item_id}/validate")
def validate_active_learning(item_id: str, payload: ActiveLearningValidationRequest) -> dict:
    item = validate_active_learning_item(
        item_id=item_id,
        validated_label=payload.validated_label,
        validator_note=payload.validator_note,
        status=payload.status,
    )
    if not item:
        raise HTTPException(status_code=404, detail="active learning item not found")
    return item
