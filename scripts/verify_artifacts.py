from pathlib import Path

REQUIRED_ARTIFACTS = [
    Path("models/model_v1.pt"),
    Path("models/label_map.json"),
    Path("models/model_v1_metadata.json"),
    Path("reports/metrics_v1.json"),
    Path("reports/confusion_matrix_v1.png"),
    Path("reports/classification_report_v1.csv"),
]


def main() -> int:
    missing = [path for path in REQUIRED_ARTIFACTS if not path.exists()]

    if missing:
        print("Artifact verification failed. Missing required files:")
        for path in missing:
            print(f"- {path}")
        print("\nTrain on Kaggle, download agrimlops_artifacts.zip, extract it to the repo root, then run this script again.")
        return 1

    print("Artifact verification passed. Required Kaggle artifacts are present:")
    for path in REQUIRED_ARTIFACTS:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"- {path} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
