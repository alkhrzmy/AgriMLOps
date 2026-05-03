import argparse
from pathlib import Path


def required_artifacts(version: str) -> list[Path]:
    return [
        Path(f"models/model_{version}.pt"),
        Path("models/label_map.json"),
        Path(f"models/model_{version}_metadata.json"),
        Path(f"reports/metrics_{version}.json"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify required AgriMLOps model artifacts.")
    parser.add_argument("--version", default="v1", choices=["v1", "v2"])
    args = parser.parse_args()

    required_files = required_artifacts(args.version)
    missing = [path for path in required_files if not path.exists()]

    if missing:
        print(f"Artifact verification failed for {args.version}. Missing required files:")
        for path in missing:
            print(f"- {path}")
        print("\nTrain on Kaggle, download agrimlops_artifacts.zip, extract it to the repo root, then run this script again.")
        return 1

    print(f"Artifact verification passed for {args.version}. Required Kaggle artifacts are present:")
    for path in required_files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"- {path} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
