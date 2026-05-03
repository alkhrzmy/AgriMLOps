import json
from pathlib import Path


def load_metadata(version: str) -> dict | None:
    path = Path(f"models/model_{version}_metadata.json")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    metadata["_path"] = str(path)
    return metadata


def metric_value(metadata: dict | None, key: str) -> float | None:
    if not metadata:
        return None
    value = metadata.get(key)
    return float(value) if value is not None else None


def format_metric(value: float | None) -> str:
    return "missing" if value is None else f"{value:.4f}"


def main() -> int:
    v1 = load_metadata("v1")
    v2 = load_metadata("v2")

    print("Model comparison")
    print("| version | status | accuracy | macro_f1 | metadata |")
    print("| --- | --- | ---: | ---: | --- |")
    for version, metadata in [("v1", v1), ("v2", v2)]:
        status = "available" if metadata else "missing"
        print(
            f"| {version} | {status} | {format_metric(metric_value(metadata, 'accuracy'))} | "
            f"{format_metric(metric_value(metadata, 'macro_f1'))} | {metadata.get('_path') if metadata else '-'} |"
        )

    if not v1:
        print("Recommendation: v1 metadata is missing. Verify current production artifacts first.")
        return 1

    if not v2:
        print("Recommendation: keep v1. v2 candidate metadata is not available yet.")
        return 0

    v1_accuracy = metric_value(v1, "accuracy") or 0.0
    v1_macro_f1 = metric_value(v1, "macro_f1") or 0.0
    v2_accuracy = metric_value(v2, "accuracy") or 0.0
    v2_macro_f1 = metric_value(v2, "macro_f1") or 0.0
    accuracy_drop = v1_accuracy - v2_accuracy

    if v2_macro_f1 >= v1_macro_f1 and accuracy_drop <= 0.02:
        print("Recommendation: promote v2. macro_f1 is not worse and accuracy drop is within 2 percentage points.")
    else:
        print("Recommendation: keep v1. v2 does not meet promotion criteria.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
