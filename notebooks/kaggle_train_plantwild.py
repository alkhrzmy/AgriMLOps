import json
import math
import random
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "huggingface_hub", "scikit-learn", "seaborn"])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

SEED = 42
REPO_ID = "uqtwei2/PlantWild"
DATA_DIR = Path("/kaggle/working/plantwild")
EXTRACT_DIR = Path("/kaggle/working/plantwild_extracted")
ARTIFACT_DIR = Path("/kaggle/working/agrimlops_artifacts")
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"
MODEL_NAME = "tf_efficientnetv2_b0"
ARCHIVE_FILENAMES = ["plantwild.zip"]
HF_ALLOW_PATTERNS = ARCHIVE_FILENAMES + ["README.md", ".gitattributes"]
TOP_K_CLASSES = 15
INPUT_SIZE = 224
EPOCHS = 8
BATCH_SIZE = 32
LR = 3e-4
MIN_IMAGES_PER_CLASS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

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
        suffix = "/" if path.is_dir() else f" ({path.stat().st_size / (1024 * 1024):.2f} MB)"
        print(f"{indent}- {path.name}{suffix}")
        shown += 1
        if shown >= max_items:
            print("...")
            break


def extract_archive(archive_path, extract_dir):
    archive_path = Path(archive_path)
    target_dir = Path(extract_dir) / archive_path.stem
    if target_dir.exists() and any(target_dir.rglob("*")):
        print(f"Using existing extracted archive: {target_dir}")
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} -> {target_dir}")
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(target_dir)
    elif archive_path.suffix.lower() in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        with tarfile.open(archive_path, "r:*") as tar_file:
            tar_file.extractall(target_dir)
    else:
        print(f"Skipped unsupported archive: {archive_path}")
    return target_dir


print_tree(DATA_DIR)

selected_archives = []
for archive_name in ARCHIVE_FILENAMES:
    archive_path = DATA_DIR / archive_name
    if archive_path.exists():
        selected_archives.append(archive_path)

if not selected_archives:
    selected_archives = [
        path
        for path in DATA_DIR.rglob("*")
        if path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}
    ]

scan_dirs = [DATA_DIR]
for archive_path in selected_archives:
    scan_dirs.append(extract_archive(archive_path, EXTRACT_DIR))

print_tree(EXTRACT_DIR)

image_extensions = {".jpg", ".jpeg", ".png"}
image_paths = []
for scan_dir in scan_dirs:
    image_paths.extend([path for path in Path(scan_dir).rglob("*") if path.suffix.lower() in image_extensions])
image_paths = sorted(set(image_paths))
print(f"Found {len(image_paths)} image files")

records = []
for image_path in image_paths:
    records.append(
        {
            "image_path": str(image_path),
            "filename": image_path.name,
            "parent_folder": image_path.parent.name,
            "label_candidate": image_path.parent.name,
        }
    )

raw_index = pd.DataFrame(records, columns=["image_path", "filename", "parent_folder", "label_candidate"])
raw_index.to_csv(REPORT_DIR / "raw_index.csv", index=False)

if raw_index.empty:
    raise RuntimeError(
        "No image files found after downloading and extracting PlantWild archives. "
        "Check DATA_DIR, EXTRACT_DIR, and ARCHIVE_FILENAMES. "
        "For PlantWild v2, try ARCHIVE_FILENAMES = ['plantwild_v2.zip']."
    )

class_counts = raw_index["label_candidate"].value_counts()
eligible_classes = class_counts[class_counts >= MIN_IMAGES_PER_CLASS].head(TOP_K_CLASSES).index.tolist()
if len(eligible_classes) < 2:
    eligible_classes = class_counts.head(TOP_K_CLASSES).index.tolist()

subset = raw_index[raw_index["label_candidate"].isin(eligible_classes)].copy()
subset = subset.rename(columns={"label_candidate": "label"})

valid_rows = []
for row in tqdm(subset.to_dict("records"), desc="Validating images"):
    try:
        with Image.open(row["image_path"]) as image:
            image.verify()
        valid_rows.append(row)
    except (UnidentifiedImageError, OSError, ValueError):
        pass

subset = pd.DataFrame(valid_rows)
labels = sorted(subset["label"].unique().tolist())
label_to_id = {label: idx for idx, label in enumerate(labels)}
id_to_label = {str(idx): label for label, idx in label_to_id.items()}
subset["label_id"] = subset["label"].map(label_to_id)

train_df, temp_df = train_test_split(
    subset,
    test_size=0.30,
    random_state=SEED,
    stratify=subset["label_id"],
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label_id"],
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()
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
plt.title("PlantWild Subset Class Distribution")
plt.tight_layout()
plt.savefig(REPORT_DIR / "class_distribution.png", dpi=160)
plt.close()

with open(MODEL_DIR / "label_map.json", "w", encoding="utf-8") as file:
    json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, file, indent=2)

train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PlantWildDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(row["label_id"])


train_loader = DataLoader(
    PlantWildDataset(train_df, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    PlantWildDataset(val_df, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    PlantWildDataset(test_df, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(labels))
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def evaluate_model(data_loader):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels_batch in data_loader:
            images = images.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            losses.append(loss.item())
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels_batch.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


best_val_f1 = -1.0
history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for images, labels_batch in progress:
        images = images.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
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
    }
    history.append(epoch_record)
    print(epoch_record)

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        torch.save(model.state_dict(), MODEL_DIR / "model_v1.pt")
        print(f"Saved best model with val_macro_f1={best_val_f1:.4f}")

model.load_state_dict(torch.load(MODEL_DIR / "model_v1.pt", map_location=DEVICE))
test_metrics = evaluate_model(test_loader)

metrics = {
    "accuracy": test_metrics["accuracy"],
    "macro_precision": test_metrics["macro_precision"],
    "macro_recall": test_metrics["macro_recall"],
    "macro_f1": test_metrics["macro_f1"],
    "best_val_macro_f1": float(best_val_f1),
    "history": history,
}
with open(REPORT_DIR / "metrics_v1.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

report_df = pd.DataFrame(classification_report(test_metrics["y_true"], test_metrics["y_pred"], target_names=labels, output_dict=True, zero_division=0)).transpose()
report_df.to_csv(REPORT_DIR / "classification_report_v1.csv")

cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=list(range(len(labels))))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - EfficientNetV2-B0 PlantWild")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix_v1.png", dpi=160)
plt.close()

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

metadata = {
    "model_version": "v1",
    "model_name": MODEL_NAME,
    "dataset": "uqtwei2/PlantWild subset",
    "num_classes": len(labels),
    "input_size": INPUT_SIZE,
    "accuracy": test_metrics["accuracy"],
    "macro_f1": test_metrics["macro_f1"],
    "created_at": datetime.now(timezone.utc).isoformat(),
    "notes": "trained on Kaggle GPU",
}
with open(MODEL_DIR / "model_v1_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

shutil.make_archive("/kaggle/working/agrimlops_artifacts", "zip", str(ARTIFACT_DIR))
print("Created /kaggle/working/agrimlops_artifacts.zip")
print(json.dumps(metadata, indent=2))
