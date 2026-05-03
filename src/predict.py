import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import timm
import torch
from PIL import Image
from torchvision import transforms

CURRENT_MODEL_VERSION = os.getenv("CURRENT_MODEL_VERSION", "v1")
MODEL_PATH = Path(os.getenv("MODEL_PATH", f"models/model_{CURRENT_MODEL_VERSION}.pt"))
LABEL_MAP_PATH = Path(os.getenv("LABEL_MAP_PATH", "models/label_map.json"))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", f"models/model_{CURRENT_MODEL_VERSION}_metadata.json"))
DEFAULT_MODEL_NAME = "tf_efficientnetv2_b0"
DEFAULT_INPUT_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.70


class ModelArtifactNotFoundError(FileNotFoundError):
    pass


def _missing_artifact_message(path: Path) -> str:
    return f"Model artifact not found: {path}. Please train on Kaggle and place the model artifact in models/."


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise ModelArtifactNotFoundError(_missing_artifact_message(path))
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_label_map(label_map: dict) -> dict[int, str]:
    if "id_to_label" in label_map:
        return {int(key): value for key, value in label_map["id_to_label"].items()}
    if "label_to_id" in label_map:
        return {int(value): key for key, value in label_map["label_to_id"].items()}
    return {int(key): value for key, value in label_map.items()}


def _build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _open_image(image_path_or_bytes) -> Image.Image:
    if isinstance(image_path_or_bytes, (str, Path)):
        return Image.open(image_path_or_bytes).convert("RGB")
    if isinstance(image_path_or_bytes, bytes):
        return Image.open(BytesIO(image_path_or_bytes)).convert("RGB")
    if hasattr(image_path_or_bytes, "read"):
        stream: BinaryIO = image_path_or_bytes
        return Image.open(stream).convert("RGB")
    raise TypeError("image_path_or_bytes must be a path, bytes, or file-like object.")


@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise ModelArtifactNotFoundError(_missing_artifact_message(MODEL_PATH))

    label_map = _normalize_label_map(_load_json(LABEL_MAP_PATH))
    metadata = _load_json(METADATA_PATH)
    model_name = metadata.get("model_name", DEFAULT_MODEL_NAME)
    input_size = int(metadata.get("input_size", DEFAULT_INPUT_SIZE))
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "label_map": label_map,
        "metadata": metadata,
        "transform": _build_transform(input_size),
        "device": device,
    }


def get_model_status() -> dict:
    artifacts_exist = MODEL_PATH.exists() and LABEL_MAP_PATH.exists() and METADATA_PATH.exists()
    metadata = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    return {
        "model_loaded": artifacts_exist,
        "current_model_version": CURRENT_MODEL_VERSION,
        "model_version": metadata.get("model_version", CURRENT_MODEL_VERSION),
        "model_name": metadata.get("model_name"),
        "model_path": str(MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
    }


def get_recommendation(label: str) -> str:
    recommendations = {
        "healthy": "Tanaman terlihat sehat. Lanjutkan pemantauan rutin, jaga sanitasi kebun, dan hindari penyiraman berlebih.",
        "rust": "Indikasi karat daun. Pisahkan daun bergejala, tingkatkan sirkulasi udara, dan konsultasikan fungisida yang sesuai.",
        "blight": "Indikasi hawar. Buang bagian tanaman terinfeksi, hindari daun terlalu basah, dan lakukan monitoring penyebaran.",
        "spot": "Indikasi bercak daun. Kurangi kelembapan berlebih, bersihkan sisa tanaman sakit, dan pantau daun sekitar.",
        "mildew": "Indikasi embun tepung/bulu. Tingkatkan jarak tanam dan sirkulasi udara, kurangi kelembapan, dan konsultasi penyuluh.",
    }
    lower_label = label.lower()
    for keyword, recommendation in recommendations.items():
        if keyword in lower_label:
            return recommendation
    return "Isolasi tanaman, ambil foto ulang dari beberapa sudut, konsultasi penyuluh, hindari penyiraman berlebih, dan periksa penyebaran gejala."


def predict_image(image_path_or_bytes) -> dict:
    bundle = load_model_bundle()
    image = _open_image(image_path_or_bytes)
    tensor = bundle["transform"](image).unsqueeze(0).to(bundle["device"])

    with torch.no_grad():
        logits = bundle["model"](tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_probabilities, top_indices = torch.topk(probabilities, k=min(3, probabilities.numel()))

    top_k = []
    for confidence, index in zip(top_probabilities.cpu().tolist(), top_indices.cpu().tolist()):
        top_k.append(
            {
                "label": bundle["label_map"][int(index)],
                "confidence": float(confidence),
            }
        )

    predicted_label = top_k[0]["label"]
    confidence = top_k[0]["confidence"]

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top_k": top_k,
        "recommendation": get_recommendation(predicted_label),
        "needs_review": confidence < LOW_CONFIDENCE_THRESHOLD,
    }
