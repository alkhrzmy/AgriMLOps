import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONTROLLED_INCORRECT_COMMENT = "controlled validation from labeled external/holdout dataset"
CONTROLLED_LOW_CONFIDENCE_COMMENT = "low-confidence correct prediction from controlled validation dataset"
CONTROLLED_VALIDATOR_NOTE = "validated from labeled controlled feedback dataset"


def load_valid_labels(label_map_path: Path) -> set[str]:
    with label_map_path.open("r", encoding="utf-8") as file:
        label_map = json.load(file)
    if "label_to_id" in label_map:
        return set(label_map["label_to_id"].keys())
    if "id_to_label" in label_map:
        return set(label_map["id_to_label"].values())
    return {str(value) for value in label_map.values()}


def iter_labeled_images(dataset_root: Path, valid_labels: set[str]) -> tuple[list[tuple[Path, str]], int]:
    images = []
    skipped_label_not_in_label_map = 0
    for class_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        ground_truth = class_dir.name
        if ground_truth not in valid_labels:
            skipped_label_not_in_label_map += sum(
                1 for image_path in class_dir.rglob("*") if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
            )
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((image_path, ground_truth))
    return images, skipped_label_not_in_label_map


def post_predict(api_base_url: str, image_path: Path) -> dict:
    with image_path.open("rb") as file:
        response = requests.post(
            f"{api_base_url}/predict",
            files={"file": (image_path.name, file, "image/jpeg")},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def post_feedback(api_base_url: str, prediction_id: str, user_feedback: str, suggested_label: str | None, comment: str | None) -> dict:
    response = requests.post(
        f"{api_base_url}/feedback",
        json={
            "prediction_id": prediction_id,
            "user_feedback": user_feedback,
            "suggested_label": suggested_label,
            "comment": comment,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_pending_queue(api_base_url: str) -> list[dict]:
    response = requests.get(f"{api_base_url}/active-learning/queue", params={"status": "pending"}, timeout=30)
    response.raise_for_status()
    return response.json().get("items", [])


def validate_queue_item(api_base_url: str, item_id: str, ground_truth: str) -> dict:
    response = requests.post(
        f"{api_base_url}/active-learning/{item_id}/validate",
        json={
            "validated_label": ground_truth,
            "status": "validated",
            "validator_note": CONTROLLED_VALIDATOR_NOTE,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def write_report(report: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled active learning simulation from a labeled folder dataset.")
    parser.add_argument("--dataset-root", required=True, help="Path to labeled folder dataset: dataset_root/class_name/image.jpg")
    parser.add_argument("--api-base-url", default="http://159.65.139.148:8000")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--label-map", default="models/label_map.json")
    parser.add_argument("--report-path", default="reports/controlled_feedback_simulation.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    label_map_path = Path(args.label_map)
    report_path = Path(args.report_path)
    api_base_url = args.api_base_url.rstrip("/")

    if not dataset_root.exists() or not dataset_root.is_dir():
        print(f"Dataset root not found or not a directory: {dataset_root}")
        return 1
    if not label_map_path.exists():
        print(f"Label map not found: {label_map_path}")
        return 1

    valid_labels = load_valid_labels(label_map_path)
    images, skipped_label_not_in_label_map = iter_labeled_images(dataset_root, valid_labels)
    selected_images = images[: args.max_images]

    report = {
        "total_images_seen": len(images) + skipped_label_not_in_label_map,
        "total_processed": 0,
        "skipped_label_not_in_label_map": skipped_label_not_in_label_map,
        "correct_high_confidence": 0,
        "incorrect_validated": 0,
        "low_confidence_validated": 0,
        "failed_requests": 0,
        "api_base_url": api_base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
    }

    print(f"Valid labels loaded: {len(valid_labels)}")
    print(f"Images found with valid labels: {len(images)}")
    print(f"Images skipped due to labels not in label_map: {skipped_label_not_in_label_map}")
    print(f"Images selected for simulation: {len(selected_images)}")

    for image_path, ground_truth in selected_images:
        print(f"Processing {image_path} | ground_truth={ground_truth}")
        if args.dry_run:
            report["total_processed"] += 1
            continue

        try:
            prediction = post_predict(api_base_url, image_path)
            prediction_id = prediction["prediction_id"]
            predicted_label = prediction["predicted_label"]
            confidence = float(prediction["confidence"])

            if predicted_label == ground_truth and confidence >= args.confidence_threshold:
                post_feedback(api_base_url, prediction_id, "correct", None, None)
                report["correct_high_confidence"] += 1
            elif predicted_label != ground_truth:
                post_feedback(api_base_url, prediction_id, "incorrect", ground_truth, CONTROLLED_INCORRECT_COMMENT)
                queue_items = get_pending_queue(api_base_url)
                matching_item = next((item for item in queue_items if item.get("prediction_id") == prediction_id), None)
                if matching_item:
                    validate_queue_item(api_base_url, matching_item["id"], ground_truth)
                    report["incorrect_validated"] += 1
                else:
                    report["failed_requests"] += 1
                    print(f"Pending queue item not found for prediction_id={prediction_id}")
            else:
                post_feedback(api_base_url, prediction_id, "unsure", ground_truth, CONTROLLED_LOW_CONFIDENCE_COMMENT)
                queue_items = get_pending_queue(api_base_url)
                matching_item = next((item for item in queue_items if item.get("prediction_id") == prediction_id), None)
                if matching_item:
                    validate_queue_item(api_base_url, matching_item["id"], ground_truth)
                    report["low_confidence_validated"] += 1
                else:
                    report["failed_requests"] += 1
                    print(f"Pending queue item not found for prediction_id={prediction_id}")

            report["total_processed"] += 1
        except requests.RequestException as exc:
            report["failed_requests"] += 1
            print(f"Request failed for {image_path}: {exc}")
        except (KeyError, ValueError) as exc:
            report["failed_requests"] += 1
            print(f"Unexpected API response for {image_path}: {exc}")

    write_report(report, report_path)
    print("Controlled feedback simulation summary")
    print(json.dumps(report, indent=2))
    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
