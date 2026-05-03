import argparse
import shutil
from pathlib import Path

REQUIRED_RELATIVE_PATHS = [
    Path("models/model_v1.pt"),
    Path("models/label_map.json"),
    Path("models/model_v1_metadata.json"),
    Path("reports/metrics_v1.json"),
    Path("reports/confusion_matrix_v1.png"),
    Path("reports/classification_report_v1.csv"),
    Path("reports/class_distribution.png"),
    Path("reports/sample_predictions.png"),
]


def copy_artifacts(source_dir: Path, target_dir: Path) -> None:
    for relative_path in REQUIRED_RELATIVE_PATHS:
        source_path = source_dir / relative_path
        target_path = target_dir / relative_path
        if source_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            print(f"Copied {source_path} -> {target_path}")
        else:
            print(f"Skipped missing optional/source artifact: {source_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy Kaggle artifacts into the local repo structure.")
    parser.add_argument("--source_dir", default="agrimlops_artifacts", help="Directory containing Kaggle artifact folders.")
    parser.add_argument("--target_dir", default=".", help="Repo root target directory.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return 1

    copy_artifacts(source_dir, target_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
