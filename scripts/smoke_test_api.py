import argparse
from pathlib import Path

import requests


def print_result(name: str, response: requests.Response) -> None:
    print(f"{name}: {response.status_code}")
    try:
        print(response.json())
    except ValueError:
        print(response.text[:500])


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test AgriMLOps FastAPI endpoints.")
    parser.add_argument("--api_base_url", default="http://localhost:8000")
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    base_url = args.api_base_url.rstrip("/")

    health = requests.get(f"{base_url}/health", timeout=20)
    print_result("GET /health", health)
    health.raise_for_status()

    model = requests.get(f"{base_url}/model/current", timeout=20)
    print_result("GET /model/current", model)
    model.raise_for_status()

    image_path = Path(args.image) if args.image else None
    if image_path is None:
        candidates = [
            Path("data/sample/sample.jpg"),
            Path("data/sample/sample.png"),
            Path("reports/sample_predictions.png"),
        ]
        image_path = next((path for path in candidates if path.exists()), None)

    if image_path and image_path.exists():
        with image_path.open("rb") as file:
            files = {"file": (image_path.name, file, "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg")}
            prediction = requests.post(f"{base_url}/predict", files=files, timeout=120)
        print_result("POST /predict", prediction)
        prediction.raise_for_status()
    else:
        print("POST /predict: skipped because no sample image was found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
