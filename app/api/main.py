from fastapi import FastAPI

from src.predict import get_model_status

app = FastAPI(
    title="AgriMLOps PlantWild API",
    description="Backend inference API for plant disease diagnosis and MLOps feedback loop.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict:
    model_status = get_model_status()
    return {
        "status": "ok",
        "service": "agrimlops-plantwild-api",
        "model_loaded": model_status["model_loaded"],
        "model_version": model_status["model_version"],
    }


@app.get("/model/current")
def current_model() -> dict:
    return get_model_status()
