import json
import random
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ==================== ENVIRONMENT DETECTION ====================
IS_KAGGLE = Path("/kaggle/input").exists()

if not IS_KAGGLE:
    def install_package(package):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            return True
        except subprocess.CalledProcessError:
            return False

    for pkg in ["timm", "huggingface_hub", "scikit-learn", "seaborn"]:
        if not install_package(pkg):
            print(f"Warning: Failed to install {pkg}")
else:
    print("Running on Kaggle - assuming packages are pre-installed or installed via !pip install")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

# ==================== CONFIGURATION ====================
SEED = 42
TOP_K_CLASSES = 15
MIN_IMAGES_PER_CLASS = 10
REPO_ID = "uqtwei2/PlantWild"

# Training hyperparameters (identical for all models)
MAX_EPOCHS = 50
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 7
EARLY_STOP_MIN_DELTA = 1e-4
T_0 = 10
T_MULT = 2

# Inference benchmark
WARMUP_BATCHES = 10
BENCHMARK_RUNS = 100

# Models to train
MODELS_CONFIG = [
    {"timm_name": "mobilenetv3_large_100", "short_name": "mobilenetv3"},
    {"timm_name": "resnet50",              "short_name": "resnet50"},
    {"timm_name": "tf_efficientnetv2_b0",  "short_name": "efficientnetv2_b0"},
]

# Paths
if IS_KAGGLE:
    DATA_DIR    = Path("/kaggle/working/plantwild")
    ARTIFACT_DIR = Path("/kaggle/working/baseline_comparison")
else:
    DATA_DIR    = Path("data/plantwild")
    ARTIFACT_DIR = Path("data/baseline_comparison")

MODEL_DIR  = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"
EXTRACT_DIR = DATA_DIR / "extracted"

ARCHIVE_FILENAMES = ["plantwild.zip", "PlantWild.zip", "plantwild_v1.zip"]

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ==================== REPRODUCIBILITY ====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ==================== DIRECTORIES ====================
for d in [DATA_DIR, ARTIFACT_DIR, MODEL_DIR, REPORT_DIR, EXTRACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("BASELINE COMPARISON TRAINING SCRIPT")
print(f"Models  : {[m['short_name'] for m in MODELS_CONFIG]}")
print(f"Classes : top-{TOP_K_CLASSES}")
print(f"Seed    : {SEED}")
print(f"Device  : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

# ==================== DATASET DOWNLOAD ====================
def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    print(f"Extracting {archive_path.name} ...")
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        import tarfile
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
    return extract_dir

def print_tree(root: Path, max_depth: int = 2, _depth: int = 0):
    if _depth > max_depth:
        return
    for item in sorted(root.iterdir()):
        print("  " * _depth + item.name + ("/" if item.is_dir() else ""))
        if item.is_dir() and _depth < max_depth:
            print_tree(item, max_depth, _depth + 1)

# Download from HuggingFace if not already present
already_downloaded = any((DATA_DIR / n).exists() for n in ARCHIVE_FILENAMES)
if not already_downloaded and not any(EXTRACT_DIR.rglob("*.jpg")):
    print(f"\nDownloading PlantWild from HuggingFace ({REPO_ID}) ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        ignore_patterns=["*.md", "*.txt"],
    )
    print("Download complete.")
else:
    print("PlantWild data already present, skipping download.")

# Extract archives
selected_archives = []
for name in ARCHIVE_FILENAMES:
    p = DATA_DIR / name
    if p.exists():
        selected_archives.append(p)

if not selected_archives:
    selected_archives = [
        p for p in DATA_DIR.rglob("*")
        if p.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}
    ]

scan_dirs = [DATA_DIR]
for arch in selected_archives:
    scan_dirs.append(extract_archive(arch, EXTRACT_DIR))

print_tree(EXTRACT_DIR)

# ==================== BUILD IMAGE INDEX ====================
print("\nBuilding image index ...")
image_extensions = {".jpg", ".jpeg", ".png"}
image_paths = []
for scan_dir in scan_dirs:
    image_paths.extend([p for p in Path(scan_dir).rglob("*") if p.suffix.lower() in image_extensions])
image_paths = sorted(set(image_paths))
print(f"Found {len(image_paths)} image files")

records = []
for ip in image_paths:
    records.append({
        "image_path":    str(ip),
        "filename":      ip.name,
        "parent_folder": ip.parent.name,
        "label_candidate": ip.parent.name,
    })

raw_index = pd.DataFrame(records, columns=["image_path", "filename", "parent_folder", "label_candidate"])
raw_index.to_csv(REPORT_DIR / "raw_index.csv", index=False)

if raw_index.empty:
    raise RuntimeError(
        "No image files found. Check DATA_DIR, EXTRACT_DIR, and ARCHIVE_FILENAMES."
    )

# ==================== SELECT TOP-K CLASSES ====================
class_counts = raw_index["label_candidate"].value_counts()
eligible_classes = (
    class_counts[class_counts >= MIN_IMAGES_PER_CLASS]
    .head(TOP_K_CLASSES)
    .index.tolist()
)
if len(eligible_classes) < 2:
    eligible_classes = class_counts.head(TOP_K_CLASSES).index.tolist()

print(f"\nSelected {len(eligible_classes)} classes: {eligible_classes}")

subset = raw_index[raw_index["label_candidate"].isin(eligible_classes)].copy()
subset = subset.rename(columns={"label_candidate": "label"})

# Validate images
print("Validating images ...")
valid_rows = []
for row in tqdm(subset.to_dict("records"), desc="Validating"):
    try:
        with Image.open(row["image_path"]) as img:
            img.verify()
        valid_rows.append(row)
    except (UnidentifiedImageError, OSError, ValueError):
        pass

subset = pd.DataFrame(valid_rows)
print(f"Valid images: {len(subset)}")

labels = sorted(subset["label"].unique().tolist())
label_to_id = {lbl: idx for idx, lbl in enumerate(labels)}
id_to_label = {str(idx): lbl for lbl, idx in label_to_id.items()}
subset["label_id"] = subset["label"].map(label_to_id)

# Save label map (shared across all models)
with open(REPORT_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)

# ==================== TRAIN / VAL / TEST SPLIT (SHARED) ====================
# This split is computed ONCE and reused for all models to ensure identical evaluation.
train_df, temp_df = train_test_split(
    subset, test_size=0.30, random_state=SEED, stratify=subset["label_id"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_id"]
)
train_df = train_df.copy().reset_index(drop=True)
val_df   = val_df.copy().reset_index(drop=True)
test_df  = test_df.copy().reset_index(drop=True)

print(f"\nSplit sizes  ->  train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

# Save split indices for reproducibility reference
train_df.to_csv(REPORT_DIR / "split_train.csv", index=False)
val_df.to_csv(REPORT_DIR / "split_val.csv", index=False)
test_df.to_csv(REPORT_DIR / "split_test.csv", index=False)

NUM_CLASSES = len(labels)

# ==================== TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ==================== DATASET ====================
class PlantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except (OSError, UnidentifiedImageError):
            img = Image.new("RGB", (224, 224), color=0)
        if self.transform:
            img = self.transform(img)
        return img, int(row["label_id"])

def make_loaders(train_df, val_df, test_df):
    train_ds = PlantDataset(train_df, train_transform)
    val_ds   = PlantDataset(val_df,   eval_transform)
    test_ds  = PlantDataset(test_df,  eval_transform)

    num_workers = 2 if IS_KAGGLE else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# ==================== MODEL BUILDER ====================
def build_model(timm_name: str, num_classes: int) -> nn.Module:
    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    return model

def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

# ==================== TRAINING UTILITIES ====================
def run_epoch(model, loader, criterion, optimizer, scaler, device, training: bool):
    model.train() if training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            with autocast():
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())
    return np.array(all_preds), np.array(all_targets)

# ==================== INFERENCE BENCHMARK ====================
def benchmark_inference(model, device, n_warmup=10, n_runs=100):
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    return float(np.mean(times)), float(np.std(times))

# ==================== PLOT UTILITIES ====================
def save_learning_curve(train_losses, val_losses, short_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses,   label="Val Loss",   linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Learning Curve — {short_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"learning_curve_{short_name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved learning_curve_{short_name}.png")

def save_confusion_matrix(cm, class_names, short_name: str):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {short_name}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"confusion_matrix_{short_name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved confusion_matrix_{short_name}.png")

# ==================== PER-MODEL TRAINING LOOP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Pre-build loaders once (shared across models)
train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df)

all_results = {}

for model_cfg in MODELS_CONFIG:
    timm_name  = model_cfg["timm_name"]
    short_name = model_cfg["short_name"]

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {short_name}  ({timm_name})")
    print("=" * 60)

    set_seed(SEED)  # Reset seed before each model for reproducibility

    # Build model
    model = build_model(timm_name, NUM_CLASSES).to(device)
    params_m = count_params(model)
    print(f"  Trainable params: {params_m:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT
    )
    scaler = GradScaler()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_loss_history = []
    val_loss_history   = []
    stopped_early = False

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, training=False)
        scheduler.step()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(
            f"  Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    --> New best val_loss: {best_val_loss:.4f} (saved checkpoint)")
        else:
            patience_counter += 1
            print(f"    --> No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch}.")
                stopped_early = True
                break

    total_epochs_trained = len(train_loss_history)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print("  Restored best model weights.")

    # ---- Save model checkpoint ----
    model_path = MODEL_DIR / f"model_{short_name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "timm_name": timm_name,
        "num_classes": NUM_CLASSES,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "epoch": total_epochs_trained,
        "best_val_loss": best_val_loss,
    }, model_path)
    print(f"  Saved model checkpoint: {model_path.name}")

    # ---- Evaluate on test set ----
    print(f"  Evaluating on test set ...")
    preds, targets = evaluate(model, test_loader, device)

    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    report_dict = classification_report(
        targets, preds, target_names=labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        targets, preds, target_names=labels, zero_division=0
    )
    cm = confusion_matrix(targets, preds)

    print(f"\n  Test Results for {short_name}:")
    print(f"    Accuracy      : {accuracy:.4f}")
    print(f"    Macro Precision: {precision:.4f}")
    print(f"    Macro Recall   : {recall:.4f}")
    print(f"    Macro F1       : {f1:.4f}")
    print(report_text)

    # ---- Inference benchmark ----
    print(f"  Running inference benchmark ({WARMUP_BATCHES} warmup + {BENCHMARK_RUNS} runs) ...")
    infer_mean_ms, infer_std_ms = benchmark_inference(
        model, device, n_warmup=WARMUP_BATCHES, n_runs=BENCHMARK_RUNS
    )
    print(f"  Inference time: {infer_mean_ms:.3f} ± {infer_std_ms:.3f} ms")

    # ---- Save per-model artifacts ----
    # metrics JSON
    metrics = {
        "model_name": timm_name,
        "short_name": short_name,
        "params_millions": round(params_m, 3),
        "accuracy": round(accuracy, 6),
        "macro_precision": round(precision, 6),
        "macro_recall": round(recall, 6),
        "macro_f1": round(f1, 6),
        "inference_time_ms_mean": round(infer_mean_ms, 4),
        "inference_time_ms_std": round(infer_std_ms, 4),
        "total_epochs_trained": total_epochs_trained,
        "stopped_early": stopped_early,
        "best_val_loss": round(best_val_loss, 6),
        "num_classes": NUM_CLASSES,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "per_class_report": report_dict,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = REPORT_DIR / f"metrics_{short_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics_{short_name}.json")

    # classification report CSV
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(REPORT_DIR / f"classification_report_{short_name}.csv")
    print(f"  Saved classification_report_{short_name}.csv")

    # confusion matrix PNG
    save_confusion_matrix(cm, labels, short_name)

    # learning curve PNG
    save_learning_curve(train_loss_history, val_loss_history, short_name)

    # Store results for comparison table
    all_results[short_name] = metrics

    # Free GPU memory before next model
    del model, optimizer, scheduler, scaler, best_state
    torch.cuda.empty_cache()
    print(f"  GPU memory released. Done with {short_name}.\n")

# ==================== COMPARISON TABLE ====================
print("\n" + "=" * 60)
print("GENERATING COMPARISON TABLE")
print("=" * 60)

comparison_rows = []
for short_name, m in all_results.items():
    comparison_rows.append({
        "model_name":           m["model_name"],
        "short_name":           short_name,
        "params_millions":      m["params_millions"],
        "accuracy":             m["accuracy"],
        "macro_precision":      m["macro_precision"],
        "macro_recall":         m["macro_recall"],
        "macro_f1":             m["macro_f1"],
        "inference_time_ms_mean": m["inference_time_ms_mean"],
        "inference_time_ms_std":  m["inference_time_ms_std"],
        "total_epochs_trained": m["total_epochs_trained"],
        "stopped_early":        m["stopped_early"],
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(ARTIFACT_DIR / "comparison_table.csv", index=False)
print("Saved comparison_table.csv")
print(comparison_df.to_string(index=False))

# Comparison summary JSON
with open(ARTIFACT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print("Saved comparison_summary.json")

# ==================== COMPARISON BAR CHART ====================
print("Generating comparison bar chart ...")

model_labels = comparison_df["short_name"].tolist()
x = np.arange(len(model_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width / 2, comparison_df["accuracy"],  width, label="Accuracy",  color="#4C72B0")
bars2 = ax.bar(x + width / 2, comparison_df["macro_f1"],  width, label="Macro F1",  color="#DD8452")

# Annotate bars
for bar in bars1:
    ax.annotate(
        f"{bar.get_height():.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4), textcoords="offset points",
        ha="center", va="bottom", fontsize=9,
    )
for bar in bars2:
    ax.annotate(
        f"{bar.get_height():.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4), textcoords="offset points",
        ha="center", va="bottom", fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title(
    f"Baseline Comparison — PlantWild top-{TOP_K_CLASSES} classes\n"
    f"(70/15/15 split, seed={SEED})"
)
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "comparison_table.png", dpi=150)
plt.close(fig)
print("Saved comparison_table.png")

# ==================== PACKAGE ARTIFACTS ====================
zip_path = shutil.make_archive(str(ARTIFACT_DIR), "zip", str(ARTIFACT_DIR))
print(f"\nArtifacts packaged: {zip_path}")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 60)
print("FINAL COMPARISON SUMMARY")
print("=" * 60)
print(comparison_df[["short_name", "params_millions", "accuracy", "macro_f1",
                      "inference_time_ms_mean", "total_epochs_trained", "stopped_early"
                      ]].to_string(index=False))
print("\nAll artifacts saved to:", ARTIFACT_DIR)
print("Done.")
