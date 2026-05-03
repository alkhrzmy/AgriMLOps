import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Float, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_PATH = Path("data/agrimlops.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")
CURRENT_MODEL_VERSION = os.getenv("CURRENT_MODEL_VERSION", "v1")
MODEL_METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", f"models/model_{CURRENT_MODEL_VERSION}_metadata.json"))

if DATABASE_URL.startswith("sqlite:///"):
    db_path = Path(DATABASE_URL.replace("sqlite:///", "", 1))
    db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(String, primary_key=True)
    image_path = Column(String, nullable=True)
    predicted_label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    top3_json = Column(Text, nullable=False)
    recommendation = Column(Text, nullable=False)
    needs_review = Column(Boolean, nullable=False, default=False)
    model_version = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id = Column(String, primary_key=True)
    prediction_id = Column(String, nullable=False, index=True)
    user_feedback = Column(String, nullable=False)
    suggested_label = Column(String, nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ActiveLearningQueue(Base):
    __tablename__ = "active_learning_queue"

    id = Column(String, primary_key=True)
    prediction_id = Column(String, nullable=False, index=True)
    image_path = Column(String, nullable=True)
    predicted_label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    reason = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending", index=True)
    validated_label = Column(String, nullable=True)
    validator_note = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    validated_at = Column(DateTime, nullable=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(String, primary_key=True)
    model_version = Column(String, nullable=False, index=True)
    model_name = Column(String, nullable=False)
    artifact_path = Column(String, nullable=False)
    accuracy = Column(Float, nullable=True)
    macro_f1 = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="deployed")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


def get_database_path() -> Path:
    if DATABASE_URL.startswith("sqlite:///"):
        return Path(DATABASE_URL.replace("sqlite:///", "", 1))
    return DATABASE_PATH


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def database_available() -> bool:
    try:
        init_db()
        with SessionLocal() as session:
            session.query(func.count(PredictionLog.id)).scalar()
        return True
    except Exception:
        return False


def _new_id() -> str:
    return str(uuid.uuid4())


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _prediction_to_dict(row: PredictionLog) -> dict:
    return {
        "id": row.id,
        "image_path": row.image_path,
        "predicted_label": row.predicted_label,
        "confidence": row.confidence,
        "top3_predictions": json.loads(row.top3_json),
        "recommendation": row.recommendation,
        "needs_review": row.needs_review,
        "model_version": row.model_version,
        "created_at": _serialize_datetime(row.created_at),
    }


def _queue_to_dict(row: ActiveLearningQueue) -> dict:
    return {
        "id": row.id,
        "prediction_id": row.prediction_id,
        "image_path": row.image_path,
        "predicted_label": row.predicted_label,
        "confidence": row.confidence,
        "reason": row.reason,
        "status": row.status,
        "validated_label": row.validated_label,
        "validator_note": row.validator_note,
        "created_at": _serialize_datetime(row.created_at),
        "validated_at": _serialize_datetime(row.validated_at),
    }


def log_prediction(
    prediction_id: str,
    image_path: str | None,
    predicted_label: str,
    confidence: float,
    top3_predictions: list[dict],
    recommendation: str,
    needs_review: bool,
    model_version: str | None,
) -> dict:
    init_db()
    with SessionLocal() as session:
        row = PredictionLog(
            id=prediction_id,
            image_path=image_path,
            predicted_label=predicted_label,
            confidence=confidence,
            top3_json=json.dumps(top3_predictions),
            recommendation=recommendation,
            needs_review=needs_review,
            model_version=model_version,
            created_at=datetime.utcnow(),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _prediction_to_dict(row)


def get_prediction(prediction_id: str) -> dict | None:
    init_db()
    with SessionLocal() as session:
        row = session.get(PredictionLog, prediction_id)
        return _prediction_to_dict(row) if row else None


def log_feedback(
    prediction_id: str,
    user_feedback: str,
    suggested_label: str | None = None,
    comment: str | None = None,
) -> dict:
    init_db()
    with SessionLocal() as session:
        row = FeedbackLog(
            id=_new_id(),
            prediction_id=prediction_id,
            user_feedback=user_feedback,
            suggested_label=suggested_label,
            comment=comment,
            created_at=datetime.utcnow(),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return {
            "id": row.id,
            "prediction_id": row.prediction_id,
            "user_feedback": row.user_feedback,
            "suggested_label": row.suggested_label,
            "comment": row.comment,
            "created_at": _serialize_datetime(row.created_at),
        }


def add_to_active_learning_queue(
    prediction_id: str,
    image_path: str | None,
    predicted_label: str,
    confidence: float,
    reason: str,
) -> dict:
    init_db()
    with SessionLocal() as session:
        existing = (
            session.query(ActiveLearningQueue)
            .filter(
                ActiveLearningQueue.prediction_id == prediction_id,
                ActiveLearningQueue.reason == reason,
                ActiveLearningQueue.status == "pending",
            )
            .first()
        )
        if existing:
            return _queue_to_dict(existing)

        row = ActiveLearningQueue(
            id=_new_id(),
            prediction_id=prediction_id,
            image_path=image_path,
            predicted_label=predicted_label,
            confidence=confidence,
            reason=reason,
            status="pending",
            created_at=datetime.utcnow(),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _queue_to_dict(row)


def get_monitoring_summary() -> dict:
    init_db()
    with SessionLocal() as session:
        predictions = session.query(PredictionLog).all()
        feedback_count = session.query(func.count(FeedbackLog.id)).scalar() or 0
        pending_count = (
            session.query(func.count(ActiveLearningQueue.id))
            .filter(ActiveLearningQueue.status == "pending")
            .scalar()
            or 0
        )

        total_predictions = len(predictions)
        confidences = [row.confidence for row in predictions]
        average_confidence = sum(confidences) / total_predictions if total_predictions else 0.0
        low_confidence_count = sum(1 for row in predictions if row.needs_review or row.confidence < 0.70)
        predictions_by_label: dict[str, int] = {}
        for row in predictions:
            predictions_by_label[row.predicted_label] = predictions_by_label.get(row.predicted_label, 0) + 1

        bins = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }
        for confidence in confidences:
            if confidence < 0.2:
                bins["0.0-0.2"] += 1
            elif confidence < 0.4:
                bins["0.2-0.4"] += 1
            elif confidence < 0.6:
                bins["0.4-0.6"] += 1
            elif confidence < 0.8:
                bins["0.6-0.8"] += 1
            else:
                bins["0.8-1.0"] += 1

        return {
            "total_predictions": total_predictions,
            "average_confidence": average_confidence,
            "low_confidence_count": low_confidence_count,
            "feedback_count": feedback_count,
            "active_learning_pending_count": pending_count,
            "predictions_by_label": predictions_by_label,
            "confidence_histogram": bins,
        }


def get_active_learning_queue(status: str = "pending") -> list[dict]:
    init_db()
    with SessionLocal() as session:
        query = session.query(ActiveLearningQueue)
        if status:
            query = query.filter(ActiveLearningQueue.status == status)
        rows = query.order_by(ActiveLearningQueue.created_at.desc()).all()
        return [_queue_to_dict(row) for row in rows]


def validate_active_learning_item(
    item_id: str,
    validated_label: str | None,
    validator_note: str | None = None,
    status: str = "validated",
) -> dict | None:
    init_db()
    with SessionLocal() as session:
        row = session.get(ActiveLearningQueue, item_id)
        if not row:
            return None
        row.status = status
        row.validated_label = validated_label
        row.validator_note = validator_note
        row.validated_at = datetime.utcnow()
        session.commit()
        session.refresh(row)
        return _queue_to_dict(row)


def register_current_model_from_metadata() -> dict | None:
    init_db()
    if not MODEL_METADATA_PATH.exists():
        return None

    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_version = metadata.get("model_version", "v1")
    with SessionLocal() as session:
        existing = (
            session.query(ModelRegistry)
            .filter(ModelRegistry.model_version == model_version, ModelRegistry.status == "deployed")
            .first()
        )
        if existing:
            return {
                "id": existing.id,
                "model_version": existing.model_version,
                "model_name": existing.model_name,
                "artifact_path": existing.artifact_path,
                "accuracy": existing.accuracy,
                "macro_f1": existing.macro_f1,
                "status": existing.status,
                "created_at": _serialize_datetime(existing.created_at),
            }

        row = ModelRegistry(
            id=_new_id(),
            model_version=model_version,
            model_name=metadata.get("model_name", "unknown"),
            artifact_path=os.getenv("MODEL_PATH", f"models/model_{CURRENT_MODEL_VERSION}.pt"),
            accuracy=metadata.get("accuracy"),
            macro_f1=metadata.get("macro_f1"),
            status="deployed",
            created_at=datetime.utcnow(),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return {
            "id": row.id,
            "model_version": row.model_version,
            "model_name": row.model_name,
            "artifact_path": row.artifact_path,
            "accuracy": row.accuracy,
            "macro_f1": row.macro_f1,
            "status": row.status,
            "created_at": _serialize_datetime(row.created_at),
        }
