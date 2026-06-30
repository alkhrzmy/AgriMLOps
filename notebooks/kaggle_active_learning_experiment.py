import json
import random
import shutil
import subprocess
import sys
import tarfile
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
IS_KAGGLE = Path("/kaggle/input").exists()
SEED = 42
TOP_K_CLASSES = 15
INPUT_SIZE = 224
BATCH_SIZE = 32
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 15
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4
N_QUERY_SIZES = [50, 100, 150, 200]
MODEL_NAME = "tf_efficientnetv2_b0"
REPO_ID = "uqtwei2/PlantWild"
MIN_IMAGES_PER_CLASS = 10
ARCHIVE_FILENAMES = ["plantwild.zip"]
HF_ALLOW_PATTERNS = ARCHIVE_FILENAMES + ["README.md", ".gitattributes"]

if IS_KAGGLE:
    BASE_MODEL_PATH = Path("/kaggle/input/agrimlops-v1-revised-artifacts/model_v1_revised.pt")
    LABEL_MAP_PATH = Path("/kaggle/input/agrimlops-v1-revised-artifacts/label_map.json")
    DATA_DIR = Path("/kaggle/working/plantwild")
    EXTRACT_DIR = Path("/kaggle/working/plantwild_extracted")
    ARTIFACT_DIR = Path("/kaggle/working/al_experiment")
else:
    BASE_MODEL_PATH = Path("models/model_v1_revised.pt")
    LABEL_MAP_PATH = Path("models/label_map.json")
    DATA_DIR = Path("data/plantwild")
    EXTRACT_DIR = Path("data/plantwild_extracted")
    ARTIFACT_DIR = Path("data/al_experiment")

REPORT_DIR = ARTIFACT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==================== DATASET LOADING ====================

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")

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
    image_paths.extend(
        [path for path in Path(scan_dir).rglob("*") if path.suffix.lower() in image_extensions]
    )
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
        "Check DATA_DIR, EXTRACT_DIR, and ARCHIVE_FILENAMES."
    )

# Select TOP_K_CLASSES by count
class_counts = raw_index["label_candidate"].value_counts()
eligible_classes = class_counts[class_counts >= MIN_IMAGES_PER_CLASS].head(TOP_K_CLASSES).index.tolist()
if len(eligible_classes) < 2:
    eligible_classes = class_counts.head(TOP_K_CLASSES).index.tolist()

subset = raw_index[raw_index["label_candidate"].isin(eligible_classes)].copy()
subset = subset.rename(columns={"label_candidate": "label"})

# Validate images with PIL verify
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

# Stratified split 70/15/15: train_pool_df / val_df / test_df
train_pool_df, temp_df = train_test_split(
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

train_pool_df = train_pool_df.copy().reset_index(drop=True)
val_df = val_df.copy().reset_index(drop=True)
test_df = test_df.copy().reset_index(drop=True)

print(f"Train pool: {len(train_pool_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Classes: {len(labels)}")

# Save label_map and dataset CSV
with open(REPORT_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)

train_pool_df["split"] = "train_pool"
val_df["split"] = "val"
test_df["split"] = "test"
dataset_df = pd.concat([train_pool_df, val_df, test_df], ignore_index=True)
dataset_df[["image_path", "label", "label_id", "split"]].to_csv(REPORT_DIR / "dataset.csv", index=False)
# Remove split column from working DataFrames
train_pool_df = train_pool_df.drop(columns=["split"]).reset_index(drop=True)
val_df = val_df.drop(columns=["split"]).reset_index(drop=True)
test_df = test_df.drop(columns=["split"]).reset_index(drop=True)

# ==================== TRANSFORMS ====================

finetune_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==================== DATASET CLASS ====================


class PlantWildDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        with Image.open(row["image_path"]) as img:
            img = img.convert("RGB")
        return self.transform(img), int(row["label_id"])


# ==================== FUNCTIONS ====================


def load_fresh_model(num_classes):
    """Load EfficientNetV2-B0 with pretrained=False, then load weights from BASE_MODEL_PATH."""
    if not BASE_MODEL_PATH.exists():
        print(f"ERROR: Base model not found at {BASE_MODEL_PATH}")
        print("Please ensure the base model artifact is available.")
        sys.exit(1)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def get_uncertainty_scores(model, pool_df):
    """
    Run inference on all samples in pool_df.
    Returns list of (pool_df_index, max_confidence) sorted ascending by confidence
    (most uncertain = lowest confidence first).
    """
    dataset = PlantWildDataset(pool_df, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    all_max_confs = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Scoring uncertainty", leave=False):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            max_conf = probs.max(dim=1).values
            all_max_confs.extend(max_conf.cpu().numpy().tolist())

    scores = [(idx, conf) for idx, conf in enumerate(all_max_confs)]
    scores.sort(key=lambda x: x[1])  # ascending: most uncertain (lowest confidence) first
    return scores


def evaluate_on_test(model, test_df):
    """
    Run inference on test_df.
    Returns dict: accuracy, macro_precision, macro_recall, macro_f1, per_class_f1.
    """
    dataset = PlantWildDataset(test_df, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, label_ids in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_true.extend(label_ids.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class F1
    _, _, per_class_f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(labels))), zero_division=0
    )
    per_class_f1 = {id_to_label[str(i)]: float(per_class_f1_arr[i]) for i in range(len(labels))}

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class_f1": per_class_f1,
    }


def benchmark_inference(model, test_df, n=100):
    """
    10 warmup runs + 100 timed single-image inference runs.
    Returns (mean_ms, std_ms).
    """
    dataset = PlantWildDataset(test_df, eval_transform)
    # Pick a single sample for repeated inference
    sample_img, _ = dataset[0]
    sample_tensor = sample_img.unsqueeze(0).to(DEVICE)

    model.eval()
    use_cuda = torch.cuda.is_available()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_tensor)
            if use_cuda:
                torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            _ = model(sample_tensor)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    return mean_ms, std_ms


def finetune(model, selected_df, val_df):
    """
    Fine-tune model on selected_df, validate on val_df.
    Uses AdamW + CosineAnnealingWarmRestarts + AMP + early stopping on val_loss.
    Returns dict with training info and loss histories.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    train_loader = DataLoader(
        PlantWildDataset(selected_df, finetune_transform),
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

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    stopped_early = False
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        # ---- Train ----
        model.train()
        epoch_train_losses = []
        progress = tqdm(train_loader, desc=f"  Finetune epoch {epoch}/{FINETUNE_EPOCHS}", leave=False)
        for images, label_ids in progress:
            images = images.to(DEVICE)
            label_ids = label_ids.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, label_ids)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_train_losses.append(loss.item())
            progress.set_postfix(loss=float(np.mean(epoch_train_losses)))
        scheduler.step()

        # ---- Validate ----
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for images, label_ids in val_loader:
                images = images.to(DEVICE)
                label_ids = label_ids.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = criterion(outputs, label_ids)
                epoch_val_losses.append(loss.item())

        train_loss = float(np.mean(epoch_train_losses))
        val_loss = float(np.mean(epoch_val_losses))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Early stopping on val_loss
        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                stopped_early = True
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    epochs_trained = len(train_loss_history)
    return {
        "epochs_trained": epochs_trained,
        "stopped_early": stopped_early,
        "best_val_loss": float(best_val_loss),
        "val_loss_history": val_loss_history,
        "train_loss_history": train_loss_history,
    }


# ==================== MAIN EXPERIMENT LOOP ====================

print("=" * 60)
print("ACTIVE LEARNING EXPERIMENT")
print("=" * 60)

results = {
    "config": {
        "model": MODEL_NAME,
        "n_query_sizes": N_QUERY_SIZES,
        "finetune_epochs": FINETUNE_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "finetune_lr": FINETUNE_LR,
        "seed": SEED,
        "train_pool_size": len(train_pool_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "num_classes": len(labels),
        "device": str(DEVICE),
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    "base_model": {},
    "experiments": [],
}

# Base model (no finetuning)
print("\n[BASE MODEL] Evaluating without finetuning...")
model = load_fresh_model(len(labels))
base_metrics = evaluate_on_test(model, test_df)
inf_mean, inf_std = benchmark_inference(model, test_df)
results["base_model"] = {
    **base_metrics,
    "inference_time_ms_mean": inf_mean,
    "inference_time_ms_std": inf_std,
}
print(f"Base model: accuracy={base_metrics['accuracy']:.4f}, macro_f1={base_metrics['macro_f1']:.4f}")
print(f"Inference: {inf_mean:.2f}ms +/- {inf_std:.2f}ms")

# For each query size
for n_query in N_QUERY_SIZES:
    print(f"\n{'=' * 60}")
    print(f"N_QUERY = {n_query}")
    print(f"{'=' * 60}")
    exp_result = {"n_query": n_query}

    # --- UNCERTAINTY SAMPLING ---
    print(f"\n[UNCERTAINTY] Selecting {n_query} most uncertain samples...")
    model_unc = load_fresh_model(len(labels))
    scores = get_uncertainty_scores(model_unc, train_pool_df)
    selected_uncertain = scores[:n_query]
    selected_indices_unc = [idx for idx, _ in selected_uncertain]
    mean_conf_unc = float(np.mean([conf for _, conf in selected_uncertain]))
    selected_df_unc = train_pool_df.iloc[selected_indices_unc].reset_index(drop=True)
    print(f"  Mean confidence of selected samples: {mean_conf_unc:.4f}")
    ft_info_unc = finetune(model_unc, selected_df_unc, val_df)
    unc_metrics = evaluate_on_test(model_unc, test_df)
    print(
        f"  Result: accuracy={unc_metrics['accuracy']:.4f}, "
        f"macro_f1={unc_metrics['macro_f1']:.4f}, "
        f"epochs={ft_info_unc['epochs_trained']}"
    )
    exp_result["uncertainty"] = {
        **unc_metrics,
        **ft_info_unc,
        "n_selected": n_query,
        "mean_confidence_selected": mean_conf_unc,
    }

    # --- RANDOM SAMPLING ---
    print(f"\n[RANDOM] Selecting {n_query} random samples...")
    model_rand = load_fresh_model(len(labels))
    rng = random.Random(SEED + n_query)
    random_indices = rng.sample(range(len(train_pool_df)), min(n_query, len(train_pool_df)))
    selected_df_rand = train_pool_df.iloc[random_indices].reset_index(drop=True)
    ft_info_rand = finetune(model_rand, selected_df_rand, val_df)
    rand_metrics = evaluate_on_test(model_rand, test_df)
    print(
        f"  Result: accuracy={rand_metrics['accuracy']:.4f}, "
        f"macro_f1={rand_metrics['macro_f1']:.4f}, "
        f"epochs={ft_info_rand['epochs_trained']}"
    )
    exp_result["random"] = {
        **rand_metrics,
        **ft_info_rand,
        "n_selected": n_query,
    }

    results["experiments"].append(exp_result)

# ==================== SAVE RESULTS AND PLOTS ====================

# Save JSON
with open(REPORT_DIR / "al_experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary table CSV
rows = []
for exp in results["experiments"]:
    rows.append({
        "n_query": exp["n_query"],
        "strategy": "uncertainty",
        "accuracy": exp["uncertainty"]["accuracy"],
        "macro_f1": exp["uncertainty"]["macro_f1"],
        "epochs_trained": exp["uncertainty"]["epochs_trained"],
        "stopped_early": exp["uncertainty"]["stopped_early"],
    })
    rows.append({
        "n_query": exp["n_query"],
        "strategy": "random",
        "accuracy": exp["random"]["accuracy"],
        "macro_f1": exp["random"]["macro_f1"],
        "epochs_trained": exp["random"]["epochs_trained"],
        "stopped_early": exp["random"]["stopped_early"],
    })
pd.DataFrame(rows).to_csv(REPORT_DIR / "al_summary_table.csv", index=False)

# F1 and Accuracy comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
n_vals = N_QUERY_SIZES
unc_f1 = [exp["uncertainty"]["macro_f1"] for exp in results["experiments"]]
rand_f1 = [exp["random"]["macro_f1"] for exp in results["experiments"]]
unc_acc = [exp["uncertainty"]["accuracy"] for exp in results["experiments"]]
rand_acc = [exp["random"]["accuracy"] for exp in results["experiments"]]
base_f1 = results["base_model"]["macro_f1"]
base_acc = results["base_model"]["accuracy"]

axes[0].plot(n_vals, unc_f1, "b-o", label="Uncertainty Sampling")
axes[0].plot(n_vals, rand_f1, "r--s", label="Random Sampling")
axes[0].axhline(y=base_f1, color="gray", linestyle=":", label=f"Base Model ({base_f1:.4f})")
axes[0].set_xlabel("Query Size (N samples)")
axes[0].set_ylabel("Macro F1")
axes[0].set_title("Active Learning: Macro F1 vs Query Size")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(n_vals, unc_acc, "b-o", label="Uncertainty Sampling")
axes[1].plot(n_vals, rand_acc, "r--s", label="Random Sampling")
axes[1].axhline(y=base_acc, color="gray", linestyle=":", label=f"Base Model ({base_acc:.4f})")
axes[1].set_xlabel("Query Size (N samples)")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Active Learning: Accuracy vs Query Size")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(REPORT_DIR / "al_comparison_plot.png", dpi=160)
plt.close()

# Loss curves per query size
for exp in results["experiments"]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n_q = exp["n_query"]

    for ax, strategy, color in zip(axes, ["uncertainty", "random"], ["blue", "red"]):
        train_hist = exp[strategy]["train_loss_history"]
        val_hist = exp[strategy]["val_loss_history"]
        epochs_range = list(range(1, len(train_hist) + 1))
        ax.plot(epochs_range, train_hist, color=color, linestyle="-", label="Train Loss")
        ax.plot(epochs_range, val_hist, color=color, linestyle="--", label="Val Loss")
        ax.set_title(f"N={n_q} - {strategy.capitalize()} Sampling")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"loss_curves_n{n_q}.png", dpi=160)
    plt.close()

# Zip artifacts
zip_path = shutil.make_archive(str(ARTIFACT_DIR), "zip", str(ARTIFACT_DIR))
print(f"\nExperiment complete. Results saved to {REPORT_DIR}")
print(f"Artifacts zipped to {zip_path}")

# Print summary table
summary_df = pd.read_csv(REPORT_DIR / "al_summary_table.csv")
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(summary_df.to_string(index=False))
print(f"\nBase model - accuracy={base_acc:.4f}, macro_f1={base_f1:.4f}")
print(f"Inference latency: {inf_mean:.2f}ms +/- {inf_std:.2f}ms")
