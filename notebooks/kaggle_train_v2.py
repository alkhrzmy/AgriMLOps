import json
import math
import random
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

# ==================== CONFIGURATION ====================
# v2: fine-tune from v1_revised with simulated feedback targeting worst classes.
# 50 misclassified samples from test set (apple scab, plum leaf, tomato late blight)
# are added to training set — simulating expert-validated active learning feedback.
IS_KAGGLE = Path("/kaggle/input").exists()
SEED = 42
TOP_K_CLASSES = 15
INPUT_SIZE = 224
BATCH_SIZE = 32
FINETUNE_LR = 1e-4
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4
N_FEEDBACK_SAMPLES = 50
# Worst classes from v1_revised — targets for simulated feedback
WORST_CLASSES = ["apple scab", "plum leaf", "tomato late blight"]
MODEL_NAME = "tf_efficientnetv2_b0"
MODEL_VERSION = "v2"
BASE_MODEL_VERSION = "v1_revised"
REPO_ID = "uqtwei2/PlantWild"
MIN_IMAGES_PER_CLASS = 10
ARCHIVE_FILENAMES = ["plantwild.zip"]
HF_ALLOW_PATTERNS = ARCHIVE_FILENAMES + ["README.md", ".gitattributes"]

if IS_KAGGLE:
    BASE_MODEL_PATH = Path("/kaggle/input/agrimlops-v1-revised-artifacts/models/model_v1_revised.pt")
    LABEL_MAP_PATH = Path("/kaggle/input/agrimlops-v1-revised-artifacts/models/label_map.json")
    DATA_DIR = Path("/kaggle/working/plantwild")
    EXTRACT_DIR = Path("/kaggle/working/plantwild_extracted")
    ARTIFACT_BASENAME = "/kaggle/working/agrimlops_artifacts_v2"
else:
    BASE_MODEL_PATH = Path("models/model_v1_revised.pt")
    LABEL_MAP_PATH = Path("models/label_map.json")
    DATA_DIR = Path("data/plantwild")
    EXTRACT_DIR = Path("data/plantwild_extracted")
    ARTIFACT_BASENAME = "data/agrimlops_artifacts_v2"

ARTIFACT_DIR = Path(ARTIFACT_BASENAME)
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Model version: {MODEL_VERSION} (fine-tune from {BASE_MODEL_VERSION})")
print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"Worst classes targeted for feedback: {WORST_CLASSES}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== DATASET LOADING ====================
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(DATA_DIR),
    allow_patterns=HF_ALLOW_PATTERNS,
)


def print_tree(root_dir, max_depth=2, max_items=80):
    root_dir = Path(root_dir)
    shown = 0
    print(f"Tree summary for {root_dir}:")
    for path in sorted(root_dir.rglob("*")):
        depth = len(path.relative_to(root_dir).parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        size_mb = path.stat().st_size / (1024 * 1024)
        suffix = "/" if path.is_dir() else f" ({size_mb:.2f} MB)"
        print(f"{indent}- {path.name}{suffix}")
        shown += 1
        if shown >= max_items:
            print("  ... (truncated)")
            break


def extract_archive(archive_path, extract_dir):
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    dest = extract_dir / archive_path.stem
    if dest.exists():
        print(f"Already extracted: {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} ...")
    if archive_path.suffix == ".zip":
        import zipfile as zf
        with zf.ZipFile(archive_path, "r") as z:
            z.extractall(dest)
    else:
        import tarfile as tf
        with tf.open(archive_path) as t:
            t.extractall(dest)
    print(f"Extracted to {dest}")
    return dest


selected_archives = [DATA_DIR / name for name in ARCHIVE_FILENAMES if (DATA_DIR / name).exists()]
if not selected_archives:
    selected_archives = [
        path for path in DATA_DIR.rglob("*")
        if path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}
    ]

scan_dirs = [DATA_DIR]
for archive_path in selected_archives:
    scan_dirs.append(extract_archive(archive_path, EXTRACT_DIR))

print_tree(EXTRACT_DIR)

image_extensions = {".jpg", ".jpeg", ".png"}
image_paths = []
for scan_dir in scan_dirs:
    image_paths.extend([p for p in Path(scan_dir).rglob("*") if p.suffix.lower() in image_extensions])
image_paths = sorted(set(image_paths))
print(f"Found {len(image_paths)} image files")

records = []
for image_path in image_paths:
    records.append({
        "image_path": str(image_path),
        "filename": image_path.name,
        "parent_folder": image_path.parent.name,
        "label_candidate": image_path.parent.name,
    })

raw_index = pd.DataFrame(records, columns=["image_path", "filename", "parent_folder", "label_candidate"])
raw_index.to_csv(REPORT_DIR / "raw_index.csv", index=False)

if raw_index.empty:
    raise RuntimeError("No image files found after downloading and extracting PlantWild archives.")

# ==================== DATASET SPLIT & LABEL MAP ====================
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    lm = json.load(f)
label_to_id = lm["label_to_id"]
id_to_label = lm["id_to_label"]
labels = [id_to_label[str(i)] for i in sorted(int(k) for k in id_to_label.keys())]
print(f"Loaded label_map with {len(labels)} classes from {LABEL_MAP_PATH}")

eligible_classes = list(label_to_id.keys())
subset = raw_index[raw_index["label_candidate"].isin(eligible_classes)].copy()
subset = subset.rename(columns={"label_candidate": "label"})

valid_rows = []
for row in tqdm(subset.to_dict("records"), desc="Validating images"):
    try:
        with Image.open(row["image_path"]) as img:
            img.verify()
        valid_rows.append(row)
    except (UnidentifiedImageError, OSError, ValueError):
        pass

subset = pd.DataFrame(valid_rows)
subset["label_id"] = subset["label"].map(label_to_id)

# Use same split as v1_revised — identical SEED ensures same train/val/test sets
train_df, temp_df = train_test_split(subset, test_size=0.30, random_state=SEED, stratify=subset["label_id"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_id"])
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

dataset_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
dataset_df = dataset_df[["image_path", "label", "label_id", "split"]]
dataset_df.to_csv(REPORT_DIR / "dataset.csv", index=False)

class_distribution = dataset_df.groupby(["split", "label"]).size().reset_index(name="count")
class_distribution.to_csv(REPORT_DIR / "class_distribution.csv", index=False)

plt.figure(figsize=(12, 8))
sns.countplot(data=dataset_df, y="label", hue="split", order=labels)
plt.title("PlantWild Subset Class Distribution (v2)")
plt.tight_layout()
plt.savefig(REPORT_DIR / "class_distribution.png", dpi=160)
plt.close()

with open(MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)

# ==================== MODEL & TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PlantWildDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        with Image.open(row["image_path"]).convert("RGB") as img:
            tensor = self.transform(img)
        return tensor, int(row["label_id"])


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 1

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(labels))
model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
print(f"Loaded base model from {BASE_MODEL_PATH}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def evaluate_model(loader):
    model.eval()
    all_preds, all_labels, all_losses = [], [], []
    with torch.no_grad():
        for images, labels_batch in loader:
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_losses.append(loss.item())
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "loss": float(np.mean(all_losses)),
        "y_true": all_labels,
        "y_pred": all_preds,
    }

# ==================== SIMULATED FEEDBACK ====================
# Simulate MLOps active learning: v1_revised running in production on unseen data (val_df).
# Expert identifies misclassified samples from worst-performing classes,
# validates correct labels, then adds them as feedback for retraining.
# Source: val_df (unseen data, simulates production) — test_df remains fully sterile.
print("Generating simulated feedback from worst-performing classes ...")
print(f"Target classes: {WORST_CLASSES}")

# Run inference on val_df to find misclassified samples
model.eval()
val_preds = []
with torch.no_grad():
    for i in tqdm(range(len(val_df)), desc="Scoring val set for feedback"):
        row = val_df.iloc[i]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
            pred_id = int(torch.argmax(model(tensor), dim=1)[0].cpu())
            val_preds.append(pred_id)
        except Exception:
            val_preds.append(-1)

val_df_scored = val_df.copy()
val_df_scored["pred_id"] = val_preds
val_df_scored["correct"] = val_df_scored["label_id"] == val_df_scored["pred_id"]

# Only consider samples from worst classes in validation set
worst_val = val_df_scored[val_df_scored["label"].isin(WORST_CLASSES)].copy()
misclassified = worst_val[~worst_val["correct"]].copy()
misclassified["source"] = "simulated_feedback_misclassified"

print(f"Misclassified val samples from worst classes: {len(misclassified)}")

# Fill to N_FEEDBACK_SAMPLES using correctly classified if needed
remaining_needed = N_FEEDBACK_SAMPLES - len(misclassified)
feedback_samples_list = [misclassified]

if remaining_needed > 0:
    correctly_classified = worst_val[worst_val["correct"]].copy()
    fill_samples = correctly_classified.sample(
        n=min(remaining_needed, len(correctly_classified)), random_state=SEED
    ).copy()
    fill_samples["source"] = "simulated_feedback_correct"
    feedback_samples_list.append(fill_samples)
    print(f"Added {len(fill_samples)} correctly-classified val samples from worst classes to reach target")

feedback_samples = pd.concat(feedback_samples_list, ignore_index=True)
feedback_save = feedback_samples[["image_path", "label", "label_id", "source", "correct"]].copy()
feedback_save.to_csv(REPORT_DIR / "simulated_feedback_v2.csv", index=False)
print(f"Total simulated feedback samples: {len(feedback_samples)}")

# Remove feedback samples from val_df to keep val set clean for evaluation
val_df = val_df[~val_df["image_path"].isin(feedback_samples["image_path"])].copy().reset_index(drop=True)
print(f"Val set after removing feedback samples: {len(val_df)}")

# Add feedback samples to train_df
fb_train = feedback_samples[["image_path", "label", "label_id"]].copy()
fb_train["split"] = "train"
train_df = pd.concat([train_df, fb_train], ignore_index=True)
print(f"Train set after adding feedback samples: {len(train_df)}")

train_loader = DataLoader(PlantWildDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(PlantWildDataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(PlantWildDataset(test_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ==================== TRAINING LOOP ====================
early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA)
best_val_f1 = 0.0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_losses = []
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}")
    for images, labels_batch in progress:
        images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())
        progress.set_postfix(loss=float(np.mean(train_losses)))
    scheduler.step()

    val_metrics = evaluate_model(val_loader)
    epoch_record = {
        "epoch": epoch,
        "train_loss": float(np.mean(train_losses)),
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    history.append(epoch_record)
    print(epoch_record)

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        torch.save(model.state_dict(), MODEL_DIR / f"model_{MODEL_VERSION}.pt")
        print(f"Saved best model val_macro_f1={best_val_f1:.4f}")

    if early_stopper(epoch_record["val_loss"], epoch):
        print(f"Early stopping at epoch {epoch}")
        break

# ==================== EVALUATION ====================
model.load_state_dict(torch.load(MODEL_DIR / f"model_{MODEL_VERSION}.pt", map_location=DEVICE))
test_metrics = evaluate_model(test_loader)

report_df = pd.DataFrame(classification_report(
    test_metrics["y_true"], test_metrics["y_pred"],
    target_names=labels, output_dict=True, zero_division=0
)).transpose()
report_df.to_csv(REPORT_DIR / f"classification_report_{MODEL_VERSION}.csv")

# ==================== PER-CLASS ANALYSIS ====================
report_dict = classification_report(
    test_metrics["y_true"], test_metrics["y_pred"],
    target_names=labels, output_dict=True
)
per_class_f1 = {label: report_dict[label]["f1-score"] for label in labels}
sorted_f1 = sorted(per_class_f1.items(), key=lambda x: x[1])
worst_classes_result = [{"class_name": k, "f1": round(v, 4)} for k, v in sorted_f1[:3]]
best_classes_result = [{"class_name": k, "f1": round(v, 4)} for k, v in sorted_f1[-3:]]
print("Worst 3 classes:", worst_classes_result)
print("Best 3 classes:", best_classes_result)

plt.figure(figsize=(10, 8))
class_names_sorted = [x[0] for x in sorted_f1]
f1_values_sorted = [x[1] for x in sorted_f1]
colors = ["#d32f2f" if v < 0.7 else "#f57c00" if v < 0.85 else "#388e3c" for v in f1_values_sorted]
plt.barh(class_names_sorted, f1_values_sorted, color=colors)
plt.xlabel("F1 Score")
plt.title(f"Per-Class F1 Score - EfficientNetV2-B0 {MODEL_VERSION}")
plt.axvline(x=np.mean(f1_values_sorted), color="blue", linestyle="--", label=f"Mean F1: {np.mean(f1_values_sorted):.3f}")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / f"per_class_f1_{MODEL_VERSION}.png", dpi=160)
plt.close()

cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=list(range(len(labels))))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - EfficientNetV2-B0 {MODEL_VERSION}")
plt.tight_layout()
plt.savefig(REPORT_DIR / f"confusion_matrix_{MODEL_VERSION}.png", dpi=160)
plt.close()

# ==================== INFERENCE BENCHMARK ====================
print("Running inference benchmark ...")
model.eval()
benchmark_images = [
    eval_transform(Image.open(test_df.iloc[i % len(test_df)]["image_path"]).convert("RGB")).unsqueeze(0).to(DEVICE)
    for i in range(100)
]
with torch.no_grad():
    for img in benchmark_images[:10]:
        _ = model(img)
if torch.cuda.is_available():
    torch.cuda.synchronize()

inference_times = []
with torch.no_grad():
    for img in benchmark_images:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

inf_mean = float(np.mean(inference_times))
inf_std = float(np.std(inference_times))
print(f"Inference time: {inf_mean:.2f} +/- {inf_std:.2f} ms")

# ==================== SAVE METRICS ====================
metrics = {
    "accuracy": test_metrics["accuracy"],
    "macro_precision": test_metrics["macro_precision"],
    "macro_recall": test_metrics["macro_recall"],
    "macro_f1": test_metrics["macro_f1"],
    "best_val_macro_f1": float(best_val_f1),
    "history": history,
    "early_stopping_epoch": early_stopper.best_epoch,
    "total_epochs_trained": epoch,
    "stopped_early": early_stopper.early_stop,
    "final_lr": float(optimizer.param_groups[0]["lr"]),
    "inference_time_ms_mean": inf_mean,
    "inference_time_ms_std": inf_std,
    "worst_classes": worst_classes_result,
    "best_classes": best_classes_result,
    "feedback_samples_added": len(feedback_samples),
    "feedback_target_classes": WORST_CLASSES,
}
with open(REPORT_DIR / f"metrics_{MODEL_VERSION}.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# ==================== VISUALIZATIONS ====================
sample_df = test_df.sample(n=min(9, len(test_df)), random_state=SEED).reset_index(drop=True)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()
model.eval()
for ax_idx, ax in enumerate(axes):
    if ax_idx >= len(sample_df):
        ax.axis("off")
        continue
    row = sample_df.iloc[ax_idx]
    image = Image.open(row["image_path"]).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs))
    pred_label = id_to_label[str(pred_id)]
    ax.imshow(image)
    ax.set_title(f"True: {row['label']}\nPred: {pred_label} ({probs[pred_id]:.2f})")
    ax.axis("off")
plt.tight_layout()
plt.savefig(REPORT_DIR / "sample_predictions.png", dpi=160)
plt.close()

# ==================== METADATA & PACKAGING ====================
dataset_summary = {
    "model_version": MODEL_VERSION,
    "plantwild_samples": int(len(subset)),
    "feedback_samples_used": int(len(feedback_samples)),
    "feedback_target_classes": WORST_CLASSES,
    "train_samples": int(len(train_df)),
    "val_samples": int(len(val_df)),
    "test_samples": int(len(test_df)),
    "num_classes": int(len(labels)),
    "classes": labels,
}
with open(REPORT_DIR / f"retraining_dataset_summary_{MODEL_VERSION}.json", "w", encoding="utf-8") as f:
    json.dump(dataset_summary, f, indent=2)

metadata = {
    "model_version": MODEL_VERSION,
    "model_name": MODEL_NAME,
    "base_model_version": BASE_MODEL_VERSION,
    "dataset": "uqtwei2/PlantWild subset + simulated feedback",
    "dataset_version": "PlantWild v1",
    "num_classes": len(labels),
    "input_size": INPUT_SIZE,
    "max_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": FINETUNE_LR,
    "optimizer": "AdamW",
    "weight_decay": 0.0001,
    "scheduler": "CosineAnnealingWarmRestarts(T_0=10, T_mult=2)",
    "split_ratio": "70/15/15",
    "seed": SEED,
    "feedback_samples_added": len(feedback_samples),
    "feedback_target_classes": WORST_CLASSES,
    "train_samples": int(len(train_df)),
    "val_samples": int(len(val_df)),
    "test_samples": int(len(test_df)),
    "accuracy": test_metrics["accuracy"],
    "macro_f1": test_metrics["macro_f1"],
    "augmentation": "RandomResizedCrop(scale=0.6-1.0), RandomHFlip, RandomVFlip, RandomRotation(15), ColorJitter(0.3), RandomGrayscale, Normalize(ImageNet)",
    "early_stopping": f"patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}, monitor=val_loss",
    "eval_transform": "Resize(256), CenterCrop(224), Normalize(ImageNet)",
    "training_platform": "Kaggle Notebook",
    "training_device": "Kaggle GPU" if torch.cuda.is_available() else "CPU",
    "framework": "PyTorch + timm",
    "pretrained_source": "v1_revised",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "notes": f"fine-tuned from {BASE_MODEL_VERSION} with {len(feedback_samples)} simulated feedback samples targeting worst classes",
}
with open(MODEL_DIR / f"model_{MODEL_VERSION}_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

zip_path = shutil.make_archive(ARTIFACT_BASENAME, "zip", str(ARTIFACT_DIR))
print(f"Created {zip_path}")
print(json.dumps(metadata, indent=2))
