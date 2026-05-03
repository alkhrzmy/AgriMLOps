import argparse
import csv
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def load_valid_labels(label_map_path: Path) -> set[str]:
    with label_map_path.open("r", encoding="utf-8") as file:
        label_map = json.load(file)
    if "label_to_id" in label_map:
        return set(label_map["label_to_id"].keys())
    if "id_to_label" in label_map:
        return set(label_map["id_to_label"].values())
    return {str(value) for value in label_map.values()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export validated active learning feedback for Kaggle retraining.")
    parser.add_argument("--database", default="data/agrimlops.db")
    parser.add_argument("--label-map", default="models/label_map.json")
    parser.add_argument("--output-dir", default="data/retraining")
    args = parser.parse_args()

    database_path = Path(args.database)
    label_map_path = Path(args.label_map)
    output_root = Path(args.output_dir)

    if not database_path.exists():
        print(f"Database not found: {database_path}")
        print("No validated feedback available for retraining yet.")
        return 0

    if not label_map_path.exists():
        print(f"Label map not found: {label_map_path}")
        return 1

    valid_labels = load_valid_labels(label_map_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_dir = output_root / f"validated_feedback_{timestamp}"
    image_dir = export_dir / "images"
    metadata_path = export_dir / "metadata.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    query = """
        SELECT
            id,
            prediction_id,
            image_path,
            predicted_label,
            confidence,
            reason,
            validated_label,
            validated_at
        FROM active_learning_queue
        WHERE status = 'validated'
          AND validated_label IS NOT NULL
          AND TRIM(validated_label) != ''
        ORDER BY validated_at ASC
    """

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query).fetchall()

    found_count = len(rows)
    exported_count = 0
    skipped_missing_file = 0
    skipped_unknown_label = 0

    metadata_rows = []
    for row in rows:
        validated_label = row["validated_label"]
        source_path = Path(row["image_path"] or "")

        if validated_label not in valid_labels:
            skipped_unknown_label += 1
            continue

        if not source_path.exists() or not source_path.is_file():
            skipped_missing_file += 1
            continue

        suffix = source_path.suffix.lower() or ".jpg"
        target_filename = f"{row['id']}{suffix}"
        target_path = image_dir / target_filename
        shutil.copy2(source_path, target_path)

        metadata_rows.append(
            {
                "image_path": f"images/{target_filename}",
                "filename": target_filename,
                "validated_label": validated_label,
                "original_prediction_id": row["prediction_id"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
                "reason": row["reason"],
                "validated_at": row["validated_at"],
            }
        )
        exported_count += 1

    fieldnames = [
        "image_path",
        "filename",
        "validated_label",
        "original_prediction_id",
        "predicted_label",
        "confidence",
        "reason",
        "validated_at",
    ]
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    archive_base = output_root / f"validated_feedback_{timestamp}"
    zip_path = shutil.make_archive(str(archive_base), "zip", str(export_dir))

    print("Validated feedback export summary")
    print(f"- validated items found: {found_count}")
    print(f"- exported items: {exported_count}")
    print(f"- skipped missing image files: {skipped_missing_file}")
    print(f"- skipped labels not in label_map: {skipped_unknown_label}")
    print(f"- export directory: {export_dir}")
    print(f"- metadata CSV: {metadata_path}")
    print(f"- ZIP: {zip_path}")

    if exported_count == 0:
        print("No validated feedback was exported. There is no retraining data yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
