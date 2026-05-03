import argparse
from pathlib import Path


def update_env_file(env_path: Path, version: str) -> bool:
    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated = False
    output_lines = []
    for line in lines:
        if line.startswith("CURRENT_MODEL_VERSION="):
            output_lines.append(f"CURRENT_MODEL_VERSION={version}")
            updated = True
        else:
            output_lines.append(line)
    if not updated:
        output_lines.append(f"CURRENT_MODEL_VERSION={version}")
    env_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a model candidate by switching CURRENT_MODEL_VERSION safely.")
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    model_path = Path(f"models/model_{args.version}.pt")
    metadata_path = Path(f"models/model_{args.version}_metadata.json")
    if not model_path.exists() or not metadata_path.exists():
        print(f"Cannot promote {args.version}: candidate artifacts are missing.")
        print(f"Required: {model_path} and {metadata_path}")
        return 1

    env_path = Path(args.env_file)
    if env_path.exists():
        existed = update_env_file(env_path, args.version)
        action = "Updated" if existed else "Added"
        print(f"{action} CURRENT_MODEL_VERSION={args.version} in {env_path}")
        print("No model files were deleted or overwritten. Restart/redeploy the API to load the promoted model.")
    else:
        print(f"No {env_path} file found. No model files were deleted or overwritten.")
        print(f"Run this command to promote {args.version}:")
        print(f'echo "CURRENT_MODEL_VERSION={args.version}" >> {env_path}')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
