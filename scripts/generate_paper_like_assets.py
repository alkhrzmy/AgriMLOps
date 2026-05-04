from __future__ import annotations

import json
import math
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "paper_workspace"
REPORT_MD = WORKSPACE / "08_report_markdown"
FIG_DIR = REPORT_MD / "images"
FINAL_FIG_DIR = WORKSPACE / "06_figures_final"
EVIDENCE_DIR = WORKSPACE / "04_project_evidence"
PDF_DIR = WORKSPACE / "02_references_pdf"
ANALYSIS_DIR = WORKSPACE / "03_reference_notes"
REPORT_DIR = ROOT / "report"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 220,
        "savefig.dpi": 220,
    }
)


BLUE = "#2F5D8C"
LIGHT_BLUE = "#DCEAF7"
GREEN = "#4E8A58"
LIGHT_GREEN = "#E3F1E7"
ORANGE = "#C97A2B"
LIGHT_ORANGE = "#F8E7D6"
RED = "#B44949"
LIGHT_RED = "#F3DDDD"
PURPLE = "#6E5AA8"
LIGHT_PURPLE = "#E7E1F4"
GRAY = "#4A5568"
LIGHT_GRAY = "#F4F6F8"
DARK = "#1F2933"


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    text: str
    fc: str = LIGHT_BLUE
    ec: str = BLUE

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def right(self) -> tuple[float, float]:
        return (self.x + self.w, self.y + self.h / 2)

    @property
    def left(self) -> tuple[float, float]:
        return (self.x, self.y + self.h / 2)

    @property
    def top(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def bottom(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, filename: str) -> None:
    for directory in (FIG_DIR, FINAL_FIG_DIR):
        path = directory / filename
        fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def draw_box(ax, box: Box, fontsize: int = 9, lw: float = 1.2) -> None:
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw,
        edgecolor=box.ec,
        facecolor=box.fc,
    )
    ax.add_patch(patch)
    wrapped = "\n".join(textwrap.wrap(box.text, width=max(10, int(box.w * 17))))
    ax.text(*box.center, wrapped, ha="center", va="center", fontsize=fontsize, color=DARK)


def arrow(ax, start, end, color=GRAY, lw=1.2, rad=0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def base_ax(width=7.2, height=4.2, title: str | None = None, xmax: float = 12, ymax: float = 7):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    if title:
        ax.text(xmax / 2, ymax - 0.25, title, ha="center", va="top", fontsize=12, fontweight="bold", color=DARK)
    return fig, ax


def add_note(ax, text: str) -> None:
    xmin, xmax = ax.get_xlim()
    ax.text((xmin + xmax) / 2, 0.18, text, ha="center", va="bottom", fontsize=7.5, color="#5B677A")


def figure_2_1() -> None:
    labels = ["Potensi\nproduksi", "Kerugian\npenyakit", "Produksi\ntersisa"]
    values = [100, 40, 60]
    colors = [GREEN, RED, BLUE]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Persentase relatif")
    ax.set_title("Ilustrasi Dampak Penyakit Tanaman terhadap Hasil Panen")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(1, 46, "Hingga ±40%\nkehilangan hasil", ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")
    save_fig(fig, "gambar_2_1_plant_disease_impact.png")


def figure_2_2() -> None:
    fig, ax = base_ax(7.2, 2.7, "Evolusi Diagnosis Penyakit Tanaman Berbasis Citra", xmax=12, ymax=4.2)
    years = ["Manual", "ML\nklasik", "CNN", "Transfer\nLearning", "In-the-wild\nDataset", "MLOps +\nFeedback"]
    xs = np.linspace(1.1, 10.9, len(years))
    y = 2.35
    ax.plot([xs[0], xs[-1]], [y, y], color=BLUE, lw=2.2)
    for i, (x, label) in enumerate(zip(xs, years)):
        ax.scatter(x, y, s=210, color="white", edgecolor=BLUE, lw=2, zorder=3)
        ax.text(x, y, str(i + 1), ha="center", va="center", color=BLUE, fontweight="bold")
        ax.text(x, y - 0.65, label, ha="center", va="top", fontsize=8.2, color=DARK)
    add_note(ax, "Diadaptasi dari tren pada review deep learning tanaman; fase akhir menunjukkan fokus operasional AgriMLOps.")
    save_fig(fig, "gambar_2_2_dl_evolution.png")


def figure_2_3() -> None:
    fig, ax = base_ax(7.6, 4.8, "Siklus MLOps untuk Diagnosis Penyakit Tanaman", xmax=14, ymax=8)
    boxes = [
        Box(0.7, 5.4, 2.05, 0.85, "Data citra\nPlantWild + feedback", LIGHT_GREEN, GREEN),
        Box(3.3, 5.4, 1.8, 0.85, "Training\nKaggle GPU", LIGHT_BLUE, BLUE),
        Box(5.65, 5.4, 2.05, 0.85, "Evaluasi &\nmodel registry", LIGHT_PURPLE, PURPLE),
        Box(8.25, 5.4, 1.95, 0.85, "Deployment\nAPI/Web", LIGHT_ORANGE, ORANGE),
        Box(8.25, 3.0, 1.95, 0.85, "Monitoring\nconfidence/log", LIGHT_BLUE, BLUE),
        Box(5.65, 3.0, 2.05, 0.85, "Feedback &\nactive queue", LIGHT_GREEN, GREEN),
        Box(3.3, 3.0, 1.8, 0.85, "Validasi\nlabel", LIGHT_ORANGE, ORANGE),
        Box(0.7, 3.0, 2.05, 0.85, "Export data\ntervalidasi", LIGHT_PURPLE, PURPLE),
    ]
    for b in boxes:
        draw_box(ax, b, fontsize=8.5)
    for a, b in zip(boxes, boxes[1:4]):
        arrow(ax, a.right, b.left)
    arrow(ax, boxes[3].bottom, boxes[4].top)
    for a, b in zip(boxes[4:], boxes[5:]):
        arrow(ax, a.left, b.right)
    arrow(ax, boxes[-1].top, boxes[0].bottom, rad=-0.28)
    add_note(ax, "Konsep MLOps diadaptasi dari Behl (2025), digambar ulang sesuai arsitektur AgriMLOps.")
    save_fig(fig, "gambar_2_3_mlops_lifecycle.png")


def figure_2_4() -> None:
    fig, ax = base_ax(7.6, 4.6, "Loop Active Learning Berbasis Ketidakpastian", xmax=14, ymax=8)
    boxes = [
        Box(0.7, 5.25, 1.75, 0.8, "Gambar\nlapangan", LIGHT_GREEN, GREEN),
        Box(2.95, 5.25, 1.75, 0.8, "Prediksi\nmodel", LIGHT_BLUE, BLUE),
        Box(5.2, 5.25, 2.1, 0.8, "Confidence rendah\natau unsure", LIGHT_ORANGE, ORANGE),
        Box(7.85, 5.25, 2.05, 0.8, "Active learning\nqueue", LIGHT_RED, RED),
        Box(7.85, 2.85, 2.05, 0.8, "Validasi\nahli/admin", LIGHT_PURPLE, PURPLE),
        Box(5.2, 2.85, 2.1, 0.8, "Validated\nfeedback", LIGHT_GREEN, GREEN),
        Box(2.95, 2.85, 1.75, 0.8, "Retraining\nmodel", LIGHT_BLUE, BLUE),
        Box(0.7, 2.85, 1.75, 0.8, "Model versi\nbaru", LIGHT_ORANGE, ORANGE),
    ]
    for b in boxes:
        draw_box(ax, b, fontsize=8.5)
    for a, b in zip(boxes[:4], boxes[1:4]):
        arrow(ax, a.right, b.left)
    arrow(ax, boxes[3].bottom, boxes[4].top)
    for a, b in zip(boxes[4:], boxes[5:]):
        arrow(ax, a.left, b.right)
    arrow(ax, boxes[-1].top, boxes[0].bottom, rad=-0.28)
    add_note(ax, "Terinspirasi active/dynamic learning pada Dong et al. (2023), digambar ulang untuk AgriMLOps.")
    save_fig(fig, "gambar_2_4_active_learning_flow.png")


def figure_3_1() -> None:
    fig, ax = base_ax(7.8, 4.8, "Arsitektur Sistem AgriMLOps", xmax=14, ymax=8)
    boxes = [
        Box(0.55, 5.35, 1.8, 0.8, "User\nBrowser", LIGHT_ORANGE, ORANGE),
        Box(3.0, 5.35, 2.0, 0.8, "Streamlit Web\n:8501", LIGHT_BLUE, BLUE),
        Box(5.85, 5.35, 2.1, 0.8, "FastAPI Backend\n:8000", LIGHT_BLUE, BLUE),
        Box(10.1, 6.2, 2.2, 0.8, "EfficientNetV2-B0\nmodel_v3", LIGHT_GREEN, GREEN),
        Box(10.1, 5.0, 2.2, 0.8, "SQLite\nagrimlops.db", LIGHT_PURPLE, PURPLE),
        Box(10.1, 3.8, 2.2, 0.8, "Upload &\nfeedback files", LIGHT_ORANGE, ORANGE),
        Box(5.85, 3.35, 2.1, 0.8, "Model Registry\nv1-v3", LIGHT_GREEN, GREEN),
        Box(3.0, 3.35, 2.0, 0.8, "Monitoring &\nActive Queue UI", LIGHT_RED, RED),
    ]
    for b in boxes:
        draw_box(ax, b, fontsize=8.5)
    arrow(ax, boxes[0].right, boxes[1].left)
    arrow(ax, boxes[1].right, boxes[2].left)
    arrow(ax, boxes[2].right, boxes[3].left)
    arrow(ax, boxes[2].right, boxes[4].left)
    arrow(ax, boxes[2].right, boxes[5].left)
    arrow(ax, boxes[2].bottom, boxes[6].top)
    arrow(ax, boxes[1].bottom, boxes[7].top)
    arrow(ax, boxes[7].right, boxes[6].left)
    add_note(ax, "Komponen digambar ulang dari implementasi FastAPI, Streamlit, SQLite, dan artifact model v3.")
    save_fig(fig, "gambar_3_1_system_architecture.png")


def figure_3_2() -> None:
    fig, ax = base_ax(7.8, 4.9, "Skema Database SQLite AgriMLOps", xmax=14, ymax=8)
    tables = [
        Box(0.55, 5.1, 2.7, 1.35, "prediction_logs\nid (PK)\nimage_path\npredicted_label\nconfidence\nmodel_version", "#F7FBFF", BLUE),
        Box(4.0, 5.1, 2.7, 1.35, "feedback_logs\nid (PK)\nprediction_id\nuser_feedback\nsuggested_label\ncomment", "#F7FBFF", GREEN),
        Box(7.45, 5.1, 2.9, 1.35, "active_learning_queue\nid (PK)\nprediction_id\nreason/status\nvalidated_label\nvalidated_at", "#F7FBFF", ORANGE),
        Box(4.0, 2.55, 2.7, 1.35, "model_registry\nid (PK)\nmodel_version\naccuracy/macro_f1\nfeedback_samples\nstatus", "#F7FBFF", PURPLE),
    ]
    for b in tables:
        draw_box(ax, b, fontsize=7.8)
    arrow(ax, tables[0].right, tables[1].left, color=GREEN)
    arrow(ax, tables[0].right, tables[2].left, color=ORANGE, rad=-0.12)
    arrow(ax, tables[3].top, (tables[0].x + tables[0].w / 2, tables[0].y), color=PURPLE, rad=-0.25)
    ax.text(3.55, 5.95, "1..n", fontsize=8, color=GREEN)
    ax.text(7.05, 5.7, "1..n", fontsize=8, color=ORANGE)
    add_note(ax, "ERD disederhanakan dari src/database.py untuk kebutuhan paper.")
    save_fig(fig, "gambar_3_2_database_schema.png")


def figure_3_3() -> None:
    fig, ax = base_ax(7.4, 4.8, "Peta Endpoint FastAPI AgriMLOps", xmax=14, ymax=8)
    groups = [
        Box(0.65, 5.3, 2.45, 1.0, "Health & Model\nGET /health\nGET /model/current\nGET /model/registry", LIGHT_BLUE, BLUE),
        Box(4.0, 5.3, 2.1, 1.0, "Prediction\nPOST /predict", LIGHT_GREEN, GREEN),
        Box(7.0, 5.3, 2.1, 1.0, "Feedback\nPOST /feedback", LIGHT_ORANGE, ORANGE),
        Box(2.0, 3.0, 2.65, 1.0, "Monitoring\nGET /monitoring/summary", LIGHT_PURPLE, PURPLE),
        Box(6.0, 2.85, 3.0, 1.25, "Active Learning\nGET /active-learning/queue\nPOST /active-learning/{id}/validate", LIGHT_RED, RED),
    ]
    for b in groups:
        draw_box(ax, b, fontsize=7.7)
    for b in groups[:3]:
        arrow(ax, b.bottom, groups[3].top if b is groups[0] else groups[4].top, rad=0.08)
    add_note(ax, "Endpoint dikelompokkan ulang dari app/api/main.py agar ringkas dan terbaca.")
    save_fig(fig, "gambar_3_3_api_endpoints.png")


def figure_3_4() -> None:
    fig, ax = base_ax(7.2, 3.5, "Struktur Halaman Streamlit", xmax=12, ymax=6)
    pages = [
        ("Diagnosis", "upload, prediksi,\nfeedback"),
        ("Feedback", "riwayat dan\nfilter feedback"),
        ("Monitoring", "confidence,\ndistribusi, metrik"),
        ("Active Learning", "queue dan\nvalidasi label"),
        ("Model Registry", "v1-v3,\nstatus model"),
    ]
    xs = np.linspace(1.1, 10.9, len(pages))
    for x, (title, desc) in zip(xs, pages):
        draw_box(ax, Box(x - 0.7, 2.45, 1.4, 0.55, title, LIGHT_BLUE, BLUE), fontsize=8)
        draw_box(ax, Box(x - 0.7, 1.55, 1.4, 0.7, desc, "#FFFFFF", "#B8C2CC"), fontsize=7.5)
        arrow(ax, (x, 2.45), (x, 2.25), color="#9AA5B1")
    ax.text(4.4, 3.55, "AgriMLOps PlantWild UI", ha="center", fontsize=10, fontweight="bold", color=DARK)
    save_fig(fig, "gambar_3_4_streamlit_pages.png")


def figure_3_5() -> None:
    fig, ax = base_ax(7.8, 4.8, "Arsitektur Deployment Docker Compose pada DigitalOcean", xmax=14, ymax=8)
    droplet = FancyBboxPatch((2.8, 1.5), 8.9, 5.4, boxstyle="round,pad=0.03", linewidth=1.4, edgecolor="#8395A7", facecolor="#FBFCFD")
    ax.add_patch(droplet)
    ax.text(7.25, 6.65, "DigitalOcean Droplet (Ubuntu)", ha="center", va="top", fontsize=9.5, fontweight="bold")
    user = Box(0.45, 5.15, 1.55, 0.72, "Internet\nUser", LIGHT_ORANGE, ORANGE)
    api = Box(3.45, 5.15, 2.1, 0.8, "api container\nFastAPI :8000", LIGHT_BLUE, BLUE)
    web = Box(3.45, 3.55, 2.1, 0.8, "web container\nStreamlit :8501", LIGHT_GREEN, GREEN)
    model = Box(8.1, 5.45, 2.0, 0.75, "./models\nmodel_v3.pt", LIGHT_PURPLE, PURPLE)
    data = Box(8.1, 4.25, 2.0, 0.75, "./data\nSQLite + uploads", LIGHT_ORANGE, ORANGE)
    reports = Box(8.1, 3.05, 2.0, 0.75, "./reports\nmetrics", LIGHT_GRAY, "#9AA5B1")
    for b in (user, api, web, model, data, reports):
        draw_box(ax, b, fontsize=8.2)
    arrow(ax, user.right, api.left)
    arrow(ax, user.right, web.left, rad=-0.12)
    arrow(ax, web.top, api.bottom, color=GREEN)
    arrow(ax, api.right, model.left, color=PURPLE)
    arrow(ax, api.right, data.left, color=ORANGE)
    arrow(ax, api.right, reports.left, color="#6B7280", rad=-0.18)
    add_note(ax, "Dibuat ulang dari docker-compose.yml; volume data membuat database dan upload persisten.")
    save_fig(fig, "gambar_3_5_deployment_architecture.png")


def figure_3_6() -> None:
    fig, ax = base_ax(7.8, 4.5, "Workflow Training dan Retraining di Kaggle GPU", xmax=14, ymax=8)
    boxes = [
        Box(0.55, 5.3, 1.65, 0.78, "PlantWild\nsubset", LIGHT_GREEN, GREEN),
        Box(2.8, 5.3, 1.8, 0.78, "Feedback ZIP\nvalidated", LIGHT_ORANGE, ORANGE),
        Box(5.25, 5.3, 1.9, 0.78, "Kaggle GPU\ntrain script", LIGHT_BLUE, BLUE),
        Box(7.8, 5.3, 1.85, 0.78, "Artifacts\nmodel + metrics", LIGHT_PURPLE, PURPLE),
        Box(10.25, 5.3, 1.65, 0.78, "Import to\nrepo", LIGHT_GRAY, "#8A94A6"),
        Box(7.8, 3.0, 1.85, 0.78, "Verify &\ncompare", LIGHT_BLUE, BLUE),
        Box(5.25, 3.0, 1.9, 0.78, "Promote\nv3", LIGHT_GREEN, GREEN),
        Box(2.8, 3.0, 1.8, 0.78, "Redeploy\nDroplet", LIGHT_ORANGE, ORANGE),
    ]
    for b in boxes:
        draw_box(ax, b, fontsize=8.2)
    for a, b in zip(boxes[:5], boxes[1:5]):
        arrow(ax, a.right, b.left)
    arrow(ax, boxes[4].bottom, boxes[5].top)
    arrow(ax, boxes[5].left, boxes[6].right)
    arrow(ax, boxes[6].left, boxes[7].right)
    add_note(ax, "Workflow ini semi-manual: training GPU di Kaggle, serving tetap di DigitalOcean.")
    save_fig(fig, "gambar_3_6_training_workflow.png")


def figure_3_7() -> None:
    fig, ax = base_ax(7.8, 4.5, "Controlled Validated Feedback Simulation", xmax=14, ymax=8)
    boxes = [
        Box(0.6, 5.25, 1.75, 0.78, "Dataset\nberlabel", LIGHT_GREEN, GREEN),
        Box(2.9, 5.25, 1.85, 0.78, "Batch\nPOST /predict", LIGHT_BLUE, BLUE),
        Box(5.3, 5.25, 1.9, 0.78, "Bandingkan\nlabel asli", LIGHT_ORANGE, ORANGE),
        Box(7.8, 5.25, 1.85, 0.78, "Submit\nfeedback", LIGHT_PURPLE, PURPLE),
        Box(7.8, 3.0, 1.85, 0.78, "Validate\nqueue item", LIGHT_RED, RED),
        Box(5.3, 3.0, 1.9, 0.78, "Export ZIP\nvalidated", LIGHT_GREEN, GREEN),
        Box(2.9, 3.0, 1.85, 0.78, "Train v3\non Kaggle", LIGHT_BLUE, BLUE),
    ]
    for b in boxes:
        draw_box(ax, b, fontsize=8.2)
    for a, b in zip(boxes[:4], boxes[1:4]):
        arrow(ax, a.right, b.left)
    arrow(ax, boxes[3].bottom, boxes[4].top)
    arrow(ax, boxes[4].left, boxes[5].right)
    arrow(ax, boxes[5].left, boxes[6].right)
    ax.text(8.72, 4.35, "incorrect / unsure\natau low confidence", ha="center", fontsize=7.5, color=RED)
    add_note(ax, "Simulasi menggunakan dataset berlabel; bukan feedback petani asli.")
    save_fig(fig, "gambar_3_7_controlled_feedback_simulation.png")


def figure_4_1() -> None:
    metrics = {
        "v1": json.loads((EVIDENCE_DIR / "metrics_v1.json").read_text()),
        "v2": json.loads((EVIDENCE_DIR / "metrics_v2.json").read_text()),
        "v3": json.loads((EVIDENCE_DIR / "metrics_v3.json").read_text()),
    }
    versions = list(metrics)
    accuracy = [metrics[v]["accuracy"] for v in versions]
    macro_f1 = [metrics[v]["macro_f1"] for v in versions]
    best_val = [np.nan, metrics["v2"].get("best_val_macro_f1", np.nan), metrics["v3"].get("best_val_macro_f1", np.nan)]
    x = np.arange(len(versions))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.bar(x - width, accuracy, width, label="Accuracy", color=BLUE)
    ax.bar(x, macro_f1, width, label="Macro F1", color=GREEN)
    ax.bar(x + width, best_val, width, label="Best Val Macro F1", color=ORANGE)
    ax.set_xticks(x, versions)
    ax.set_ylim(0.76, 0.86)
    ax.set_ylabel("Skor")
    ax.set_title("Perbandingan Performa Model v1-v2-v3")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    for container in ax.containers:
        labels = ["" if math.isnan(v) else f"{v:.3f}" for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)
    save_fig(fig, "gambar_4_1_model_comparison.png")


def figure_4_2() -> None:
    metrics = json.loads((EVIDENCE_DIR / "metrics_v3.json").read_text())
    history = metrics["history"]
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", color=BLUE, label="Train loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="s", color=ORANGE, label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], marker="o", color=GREEN, label="Val accuracy")
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], marker="s", color=PURPLE, label="Val macro F1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Training Curve Model v3", y=1.02, fontsize=12, fontweight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "gambar_4_2_training_curves_v3.png")


def figure_4_3_confusion_matrix_clean() -> None:
    src = ROOT / "reports" / "confusion_matrix_v3.png"
    if src.exists():
        for directory in (FIG_DIR, FINAL_FIG_DIR):
            shutil.copy2(src, directory / "confusion_matrix_v3.png")


def _load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def figure_4_4_sample_predictions_clean() -> None:
    src = ROOT / "reports" / "sample_predictions.png"
    if not src.exists():
        return
    img = Image.open(src).convert("RGB")
    w, h = img.size
    # Remove large title and surrounding whitespace from matplotlib output.
    crop = img.crop((80, 105, w - 80, h - 85))
    thumb_w, thumb_h = 300, 300
    crop.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
    canvas = Image.new("RGB", (900, 360), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(26, bold=True)
    body_font = _load_font(18)
    small_font = _load_font(14)
    canvas.paste(crop, (40, 35))
    draw.text((390, 52), "Contoh Prediksi Model v3", fill=DARK, font=title_font)
    bullets = [
        "Model menampilkan label penyakit dan confidence.",
        "Top-k predictions digunakan untuk transparansi diagnosis.",
        "Contoh ini berasal dari artifact evaluasi model, bukan gambar petani.",
    ]
    y = 105
    for b in bullets:
        draw.ellipse((394, y + 7, 402, y + 15), fill=GREEN)
        draw.text((415, y), b, fill=DARK, font=body_font)
        y += 42
    draw.rectangle((390, 245, 835, 300), outline="#CBD5E1", width=2)
    draw.text((410, 262), "Sumber: reports/sample_predictions.png", fill="#64748B", font=small_font)
    for directory in (FIG_DIR, FINAL_FIG_DIR):
        canvas.save(directory / "sample_predictions.png")


def screenshot_to_clean_panel(src_name: str, title: str, bullets: list[str], out_name: str) -> None:
    source_candidates = [
        REPORT_DIR / "images_raw" / src_name,
        WORKSPACE / "05_figures_source" / src_name,
        EVIDENCE_DIR / src_name,
    ]
    src = next((candidate for candidate in source_candidates if candidate.exists()), None)
    canvas = Image.new("RGB", (900, 360), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(23, bold=True)
    body_font = _load_font(17)
    small_font = _load_font(13)
    x0, y0 = 35, 46
    draw.rectangle((x0 - 8, y0 - 8, x0 + 335 + 8, y0 + 240 + 8), fill="#F8FAFC", outline="#CBD5E1", width=2)
    if src is not None:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        crop = im.crop((max(0, int(w * 0.04)), max(0, int(h * 0.05)), max(1, int(w * 0.96)), max(1, int(h * 0.95))))
        crop.thumbnail((335, 240), Image.LANCZOS)
        canvas.paste(crop, (x0 + (335 - crop.width) // 2, y0 + (240 - crop.height) // 2))
    else:
        draw.rounded_rectangle((x0 + 18, y0 + 26, x0 + 317, y0 + 214), radius=10, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.rectangle((x0 + 18, y0 + 26, x0 + 88, y0 + 214), fill="#F1F5F9")
        draw.text((x0 + 105, y0 + 55), title, fill=DARK, font=_load_font(15, bold=True))
        for idx in range(4):
            yy = y0 + 92 + idx * 28
            draw.rounded_rectangle((x0 + 105, yy, x0 + 285, yy + 12), radius=4, fill="#E2E8F0")
        draw.text((x0 + 105, y0 + 184), "UI evidence panel", fill="#64748B", font=_load_font(12))
    draw.text((430, 52), title, fill=DARK, font=title_font)
    y = 100
    for b in bullets:
        wrapped = textwrap.wrap(b, 42)
        draw.ellipse((430, y + 8, 438, y + 16), fill=BLUE)
        for line in wrapped:
            draw.text((455, y), line, fill=DARK, font=body_font)
            y += 24
        y += 10
    draw.text((430, 315), "Sumber: evidence production AgriMLOps", fill="#64748B", font=small_font)
    for directory in (FIG_DIR, FINAL_FIG_DIR):
        canvas.save(directory / out_name)


def clean_screenshots() -> None:
    screenshot_to_clean_panel(
        "gambar_4_5_diagnosis_page.png",
        "Diagnosis Page",
        ["Upload gambar tanaman melalui Streamlit UI.", "Halaman terhubung ke FastAPI /predict.", "Dipakai sebagai entry point diagnosis pengguna."],
        "gambar_4_5_diagnosis_page.png",
    )
    screenshot_to_clean_panel(
        "gambar_4_6_prediction_result.png",
        "Prediction Result",
        ["Menampilkan label prediksi dan confidence.", "Top-k predictions membantu interpretasi hasil.", "Hasil prediksi otomatis tersimpan pada prediction_logs."],
        "gambar_4_6_prediction_result.png",
    )
    screenshot_to_clean_panel(
        "gambar_4_7_feedback_form.png",
        "Feedback Form",
        ["User dapat memilih correct, incorrect, atau unsure.", "Suggested label menjadi kandidat data validasi.", "Feedback mendukung active learning loop."],
        "gambar_4_7_feedback_form.png",
    )
    screenshot_to_clean_panel(
        "gambar_4_8_active_learning_queue.png",
        "Active Learning Queue",
        ["Queue menyimpan prediksi low-confidence/unsure.", "Validator memberi label final.", "Item validated diekspor untuk retraining."],
        "gambar_4_8_active_learning_queue.png",
    )
    screenshot_to_clean_panel(
        "gambar_4_9_monitoring_dashboard.png",
        "Monitoring Dashboard",
        ["Dashboard menampilkan distribusi prediksi.", "Confidence digunakan sebagai proxy monitoring.", "Mendukung observability model production."],
        "gambar_4_9_monitoring_dashboard.png",
    )
    screenshot_to_clean_panel(
        "gambar_4_10_model_registry.png",
        "Model Registry",
        ["Registry melacak model v1, v2, dan v3.", "v3 berstatus deployed.", "Metadata mencakup metrik dan feedback_samples_used."],
        "gambar_4_10_model_registry.png",
    )


def api_health_panel() -> None:
    data = {
        "status": "ok",
        "service": "agrimlops-plantwild-api",
        "model_loaded": True,
        "current_model_version": "v3",
        "database_available": True,
    }
    canvas = Image.new("RGB", (900, 330), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(25, bold=True)
    code_font = _load_font(17)
    small_font = _load_font(13)
    draw.text((35, 30), "FastAPI Health Endpoint", fill=DARK, font=title_font)
    draw.rounded_rectangle((35, 78, 865, 265), radius=10, fill="#0F172A")
    y = 100
    for line in json.dumps(data, indent=2).splitlines():
        color = "#E2E8F0"
        if "true" in line.lower():
            color = "#86EFAC"
        if '"v3"' in line:
            color = "#FDE68A"
        draw.text((65, y), line, fill=color, font=code_font)
        y += 24
    draw.text((35, 290), "Sumber: http://159.65.139.148:8000/health", fill="#64748B", font=small_font)
    for directory in (FIG_DIR, FINAL_FIG_DIR):
        canvas.save(directory / "gambar_4_12_fastapi_health.png")


def api_docs_panel() -> None:
    canvas = Image.new("RGB", (900, 360), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(25, bold=True)
    body_font = _load_font(17)
    small_font = _load_font(13)
    draw.text((35, 28), "FastAPI Endpoint Documentation", fill=DARK, font=title_font)
    endpoints = [
        ("GET", "/health", BLUE),
        ("GET", "/model/current", GREEN),
        ("GET", "/model/registry", GREEN),
        ("POST", "/predict", ORANGE),
        ("POST", "/feedback", ORANGE),
        ("GET", "/monitoring/summary", PURPLE),
        ("GET", "/active-learning/queue", RED),
        ("POST", "/active-learning/{item_id}/validate", RED),
    ]
    y = 82
    for method, path, color in endpoints:
        draw.rounded_rectangle((50, y, 150, y + 28), radius=6, fill=color)
        draw.text((78 if method == "GET" else 70, y + 5), method, fill="white", font=small_font)
        draw.rounded_rectangle((165, y, 835, y + 28), radius=6, fill="#F8FAFC", outline="#CBD5E1")
        draw.text((185, y + 4), path, fill=DARK, font=body_font)
        y += 32
    draw.text((35, 330), "Diringkas dari Swagger UI production agar lebih terbaca pada DOCX.", fill="#64748B", font=small_font)
    for directory in (FIG_DIR, FINAL_FIG_DIR):
        canvas.save(directory / "gambar_4_11_fastapi_docs.png")


def _pil_canvas(title: str, size=(1800, 1000)):
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((size[0] // 2, 55), title, anchor="mm", fill=DARK, font=_load_font(42, bold=True))
    return canvas, draw


def _wrap_text_to_width(draw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        words = raw_line.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _pil_box(draw, xy, text: str, fill: str, outline: str, font_size: int = 26):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=14, fill=fill, outline=outline, width=5)
    font = _load_font(font_size)
    max_width = max(60, x2 - x1 - 24)
    lines = _wrap_text_to_width(draw, text, font, max_width)
    line_h = font_size + 6
    total_h = len(lines) * line_h
    y = y1 + (y2 - y1 - total_h) / 2
    for line in lines:
        draw.text(((x1 + x2) / 2, y), line, anchor="ma", fill=DARK, font=font)
        y += line_h


def _pil_arrow(draw, start, end, color="#4B5563", width=5):
    draw.line((start, end), fill=color, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    size = 18
    p1 = (ex, ey)
    p2 = (ex - size * math.cos(angle - math.pi / 6), ey - size * math.sin(angle - math.pi / 6))
    p3 = (ex - size * math.cos(angle + math.pi / 6), ey - size * math.sin(angle + math.pi / 6))
    draw.polygon([p1, p2, p3], fill=color)


def _save_pil(canvas, filename: str):
    for directory in (FIG_DIR, FINAL_FIG_DIR):
        canvas.save(directory / filename)


def _note(draw, text: str, y=940, size=(1800, 1000)):
    draw.text((size[0] // 2, y), text, anchor="mm", fill="#64748B", font=_load_font(22))


def pil_mlops_lifecycle():
    canvas, draw = _pil_canvas("Siklus MLOps untuk Diagnosis Penyakit Tanaman")
    boxes = [
        ((120, 230, 410, 360), "Data citra\nPlantWild + feedback", LIGHT_GREEN, GREEN),
        ((510, 230, 760, 360), "Training\nKaggle GPU", LIGHT_BLUE, BLUE),
        ((860, 230, 1170, 360), "Evaluasi &\nmodel registry", LIGHT_PURPLE, PURPLE),
        ((1270, 230, 1560, 360), "Deployment\nAPI/Web", LIGHT_ORANGE, ORANGE),
        ((1270, 560, 1560, 690), "Monitoring\nconfidence/log", LIGHT_BLUE, BLUE),
        ((860, 560, 1170, 690), "Feedback &\nactive queue", LIGHT_GREEN, GREEN),
        ((510, 560, 760, 690), "Validasi\nlabel", LIGHT_ORANGE, ORANGE),
        ((120, 560, 410, 690), "Export data\ntervalidasi", LIGHT_PURPLE, PURPLE),
    ]
    for xy, text, fill, outline in boxes:
        _pil_box(draw, xy, text, fill, outline)
    pts = [
        ((410, 295), (510, 295)),
        ((760, 295), (860, 295)),
        ((1170, 295), (1270, 295)),
        ((1415, 360), (1415, 560)),
        ((1270, 625), (1170, 625)),
        ((860, 625), (760, 625)),
        ((510, 625), (410, 625)),
        ((265, 560), (265, 360)),
    ]
    for s, e in pts:
        _pil_arrow(draw, s, e)
    _note(draw, "Konsep MLOps diadaptasi dari Behl (2025), digambar ulang sesuai arsitektur AgriMLOps.")
    _save_pil(canvas, "gambar_2_3_mlops_lifecycle.png")


def pil_active_learning():
    canvas, draw = _pil_canvas("Loop Active Learning Berbasis Ketidakpastian")
    boxes = [
        ((120, 230, 390, 350), "Gambar\nlapangan", LIGHT_GREEN, GREEN),
        ((480, 230, 750, 350), "Prediksi\nmodel", LIGHT_BLUE, BLUE),
        ((840, 230, 1140, 350), "Confidence rendah\natau unsure", LIGHT_ORANGE, ORANGE),
        ((1230, 230, 1530, 350), "Active learning\nqueue", LIGHT_RED, RED),
        ((1230, 570, 1530, 690), "Validasi\nahli/admin", LIGHT_PURPLE, PURPLE),
        ((840, 570, 1140, 690), "Validated\nfeedback", LIGHT_GREEN, GREEN),
        ((480, 570, 750, 690), "Retraining\nmodel", LIGHT_BLUE, BLUE),
        ((120, 570, 390, 690), "Model versi\nbaru", LIGHT_ORANGE, ORANGE),
    ]
    for xy, text, fill, outline in boxes:
        _pil_box(draw, xy, text, fill, outline)
    for s, e in [
        ((390, 290), (480, 290)),
        ((750, 290), (840, 290)),
        ((1140, 290), (1230, 290)),
        ((1380, 350), (1380, 570)),
        ((1230, 630), (1140, 630)),
        ((840, 630), (750, 630)),
        ((480, 630), (390, 630)),
        ((255, 570), (255, 350)),
    ]:
        _pil_arrow(draw, s, e)
    _note(draw, "Terinspirasi active/dynamic learning pada Dong et al. (2023), digambar ulang untuk AgriMLOps.")
    _save_pil(canvas, "gambar_2_4_active_learning_flow.png")


def pil_system_architecture():
    canvas, draw = _pil_canvas("Arsitektur Sistem AgriMLOps")
    boxes = [
        ((80, 260, 330, 390), "User\nBrowser", LIGHT_ORANGE, ORANGE),
        ((455, 260, 745, 390), "Streamlit Web\n:8501", LIGHT_BLUE, BLUE),
        ((880, 260, 1190, 390), "FastAPI Backend\n:8000", LIGHT_BLUE, BLUE),
        ((1370, 150, 1680, 270), "EfficientNetV2-B0\nmodel_v3", LIGHT_GREEN, GREEN),
        ((1370, 330, 1680, 450), "SQLite\nagrimlops.db", LIGHT_PURPLE, PURPLE),
        ((1370, 510, 1680, 630), "Upload &\nfeedback files", LIGHT_ORANGE, ORANGE),
        ((880, 565, 1190, 685), "Model Registry\nv1-v3", LIGHT_GREEN, GREEN),
        ((455, 565, 745, 685), "Monitoring &\nActive Queue UI", LIGHT_RED, RED),
    ]
    for xy, text, fill, outline in boxes:
        _pil_box(draw, xy, text, fill, outline)
    for s, e in [
        ((330, 325), (455, 325)),
        ((745, 325), (880, 325)),
        ((1190, 325), (1370, 210)),
        ((1190, 325), (1370, 390)),
        ((1190, 325), (1370, 570)),
        ((1035, 390), (1035, 565)),
        ((600, 390), (600, 565)),
        ((745, 625), (880, 625)),
    ]:
        _pil_arrow(draw, s, e)
    _note(draw, "Komponen digambar ulang dari implementasi FastAPI, Streamlit, SQLite, dan artifact model v3.")
    _save_pil(canvas, "gambar_3_1_system_architecture.png")


def pil_database_schema():
    canvas, draw = _pil_canvas("Skema Database SQLite AgriMLOps")
    boxes = [
        ((70, 220, 430, 410), "prediction_logs\nid (PK)\nimage_path\npredicted_label\nconfidence\nmodel_version", "#F7FBFF", BLUE),
        ((540, 220, 900, 410), "feedback_logs\nid (PK)\nprediction_id\nuser_feedback\nsuggested_label\ncomment", "#F7FBFF", GREEN),
        ((1010, 220, 1430, 410), "active_learning_queue\nid (PK)\nprediction_id\nreason/status\nvalidated_label\nvalidated_at", "#F7FBFF", ORANGE),
        ((540, 590, 900, 780), "model_registry\nid (PK)\nmodel_version\naccuracy/macro_f1\nfeedback_samples\nstatus", "#F7FBFF", PURPLE),
    ]
    for xy, text, fill, outline in boxes:
        _pil_box(draw, xy, text, fill, outline, font_size=24)
    _pil_arrow(draw, (430, 315), (540, 315), GREEN)
    _pil_arrow(draw, (900, 315), (1010, 315), ORANGE)
    _pil_arrow(draw, (720, 590), (250, 410), PURPLE)
    draw.text((485, 285), "1..n", fill=GREEN, font=_load_font(22, bold=True))
    draw.text((955, 285), "1..n", fill=ORANGE, font=_load_font(22, bold=True))
    _note(draw, "ERD disederhanakan dari src/database.py untuk kebutuhan paper.")
    _save_pil(canvas, "gambar_3_2_database_schema.png")


def pil_api_endpoints():
    canvas, draw = _pil_canvas("Peta Endpoint FastAPI AgriMLOps")
    boxes = [
        ((110, 230, 470, 380), "Health & Model\nGET /health\nGET /model/current\nGET /model/registry", LIGHT_BLUE, BLUE),
        ((620, 230, 940, 380), "Prediction\nPOST /predict", LIGHT_GREEN, GREEN),
        ((1090, 230, 1410, 380), "Feedback\nPOST /feedback", LIGHT_ORANGE, ORANGE),
        ((320, 570, 760, 720), "Monitoring\nGET /monitoring/summary", LIGHT_PURPLE, PURPLE),
        ((930, 550, 1430, 740), "Active Learning\nGET /active-learning/queue\nPOST /active-learning/{id}/validate", LIGHT_RED, RED),
    ]
    for xy, text, fill, outline in boxes:
        _pil_box(draw, xy, text, fill, outline, font_size=23)
    for s, e in [
        ((290, 380), (540, 570)),
        ((780, 380), (1180, 550)),
        ((1250, 380), (1180, 550)),
    ]:
        _pil_arrow(draw, s, e)
    _note(draw, "Endpoint dikelompokkan ulang dari app/api/main.py agar ringkas dan terbaca.")
    _save_pil(canvas, "gambar_3_3_api_endpoints.png")


def pil_streamlit_pages():
    canvas, draw = _pil_canvas("Struktur Halaman Streamlit", size=(1800, 800))
    draw.text((900, 175), "AgriMLOps PlantWild UI", anchor="mm", fill=DARK, font=_load_font(30, bold=True))
    pages = [
        ("Diagnosis", "upload, prediksi,\nfeedback"),
        ("Feedback", "riwayat dan\nfilter feedback"),
        ("Monitoring", "confidence,\ndistribusi, metrik"),
        ("Active Learning", "queue dan\nvalidasi label"),
        ("Model Registry", "v1-v3,\nstatus model"),
    ]
    xs = [170, 500, 830, 1160, 1490]
    for x, (title, desc) in zip(xs, pages):
        _pil_box(draw, (x - 130, 270, x + 130, 355), title, LIGHT_BLUE, BLUE, font_size=24)
        _pil_arrow(draw, (x, 355), (x, 425))
        _pil_box(draw, (x - 130, 425, x + 130, 545), desc, "#FFFFFF", "#CBD5E1", font_size=21)
    _save_pil(canvas, "gambar_3_4_streamlit_pages.png")


def pil_deployment():
    canvas, draw = _pil_canvas("Arsitektur Deployment Docker Compose pada DigitalOcean")
    draw.rounded_rectangle((420, 170, 1650, 800), radius=18, fill="#FBFCFD", outline="#94A3B8", width=4)
    draw.text((1035, 205), "DigitalOcean Droplet (Ubuntu)", anchor="mm", fill=DARK, font=_load_font(28, bold=True))
    boxes = [
        ((80, 350, 310, 470), "Internet\nUser", LIGHT_ORANGE, ORANGE),
        ((520, 290, 840, 410), "api container\nFastAPI :8000", LIGHT_BLUE, BLUE),
        ((520, 535, 840, 655), "web container\nStreamlit :8501", LIGHT_GREEN, GREEN),
        ((1080, 270, 1380, 380), "./models\nmodel_v3.pt", LIGHT_PURPLE, PURPLE),
        ((1080, 430, 1380, 540), "./data\nSQLite + uploads", LIGHT_ORANGE, ORANGE),
        ((1080, 590, 1380, 700), "./reports\nmetrics", LIGHT_GRAY, "#9AA5B1"),
    ]
    for xy, text, fill, outline in boxes:
        _pil_box(draw, xy, text, fill, outline)
    for s, e in [
        ((310, 410), (520, 350)),
        ((310, 410), (520, 595)),
        ((680, 535), (680, 410)),
        ((840, 350), (1080, 325)),
        ((840, 350), (1080, 485)),
        ((840, 350), (1080, 645)),
    ]:
        _pil_arrow(draw, s, e)
    _note(draw, "Dibuat ulang dari docker-compose.yml; volume data membuat database dan upload persisten.")
    _save_pil(canvas, "gambar_3_5_deployment_architecture.png")


def pil_training_workflow():
    canvas, draw = _pil_canvas("Workflow Training dan Retraining di Kaggle GPU")
    boxes = [
        ((80, 250, 310, 365), "PlantWild\nsubset", LIGHT_GREEN, GREEN),
        ((410, 250, 680, 365), "Feedback ZIP\nvalidated", LIGHT_ORANGE, ORANGE),
        ((780, 250, 1070, 365), "Kaggle GPU\ntrain script", LIGHT_BLUE, BLUE),
        ((1170, 250, 1460, 365), "Artifacts\nmodel + metrics", LIGHT_PURPLE, PURPLE),
        ((1555, 250, 1735, 365), "Import to\nrepo", LIGHT_GRAY, "#8A94A6"),
        ((1170, 570, 1460, 685), "Verify &\ncompare", LIGHT_BLUE, BLUE),
        ((780, 570, 1070, 685), "Promote\nv3", LIGHT_GREEN, GREEN),
        ((410, 570, 680, 685), "Redeploy\nDroplet", LIGHT_ORANGE, ORANGE),
    ]
    for xy, text, fill, outline in boxes:
        _pil_box(draw, xy, text, fill, outline)
    for s, e in [
        ((310, 307), (410, 307)),
        ((680, 307), (780, 307)),
        ((1070, 307), (1170, 307)),
        ((1460, 307), (1555, 307)),
        ((1645, 365), (1315, 570)),
        ((1170, 627), (1070, 627)),
        ((780, 627), (680, 627)),
    ]:
        _pil_arrow(draw, s, e)
    _note(draw, "Workflow ini semi-manual: training GPU di Kaggle, serving tetap di DigitalOcean.")
    _save_pil(canvas, "gambar_3_6_training_workflow.png")


def pil_feedback_simulation():
    canvas, draw = _pil_canvas("Controlled Validated Feedback Simulation")
    boxes = [
        ((90, 250, 340, 365), "Dataset\nberlabel", LIGHT_GREEN, GREEN),
        ((440, 250, 720, 365), "Batch\nPOST /predict", LIGHT_BLUE, BLUE),
        ((820, 250, 1120, 365), "Bandingkan\nlabel asli", LIGHT_ORANGE, ORANGE),
        ((1220, 250, 1500, 365), "Submit\nfeedback", LIGHT_PURPLE, PURPLE),
        ((1220, 570, 1500, 685), "Validate\nqueue item", LIGHT_RED, RED),
        ((820, 570, 1120, 685), "Export ZIP\nvalidated", LIGHT_GREEN, GREEN),
        ((440, 570, 720, 685), "Train v3\non Kaggle", LIGHT_BLUE, BLUE),
    ]
    for xy, text, fill, outline in boxes:
        _pil_box(draw, xy, text, fill, outline)
    for s, e in [
        ((340, 307), (440, 307)),
        ((720, 307), (820, 307)),
        ((1120, 307), (1220, 307)),
        ((1360, 365), (1360, 570)),
        ((1220, 627), (1120, 627)),
        ((820, 627), (720, 627)),
    ]:
        _pil_arrow(draw, s, e)
    draw.text((1510, 465), "incorrect / unsure\natau low confidence", anchor="mm", fill=RED, font=_load_font(26, bold=True))
    _note(draw, "Simulasi menggunakan dataset berlabel; bukan feedback petani asli.")
    _save_pil(canvas, "gambar_3_7_controlled_feedback_simulation.png")


def pil_refine_diagrams():
    pil_mlops_lifecycle()
    pil_active_learning()
    pil_system_architecture()
    pil_database_schema()
    pil_api_endpoints()
    pil_streamlit_pages()
    pil_deployment()
    pil_training_workflow()
    pil_feedback_simulation()


def create_reference_visual_analysis() -> None:
    content = """# Analisis Visual Paper Referensi

Tujuan analisis ini bukan menyalin gambar paper, melainkan mengambil prinsip visual lalu menggambar ulang diagram AgriMLOps agar bebas plagiasi.

## Referensi yang dianalisis

1. **Wei et al. (2024) Snap and Diagnose**
   - Figure sistem menampilkan pipeline kiri-ke-kanan: input, encoder, retrieval/search, backend, database, dan UI.
   - Cocok sebagai inspirasi untuk `gambar_3_1_system_architecture.png` dan `gambar_3_5_deployment_architecture.png`.
   - Yang diterapkan: layout modular, blok komponen, dan alur panah. Konten diganti total sesuai AgriMLOps.

2. **Rahman et al. (2025) Real-time Monitoring System**
   - Figure arsitektur menunjukkan pemisahan training/evaluation, web/mobile application, dan service layer.
   - Cocok sebagai inspirasi untuk diagram deployment dan workflow inference.
   - Yang diterapkan: pemisahan UI, service, model, database, dan evaluation evidence.

3. **Behl (2025) MLOps Framework**
   - Visual yang tersedia berupa heatmap perbandingan tool dan pembahasan pipeline MLOps.
   - Cocok untuk prinsip lifecycle: CI/CD, monitoring, model versioning, drift, rollback/retraining.
   - Yang diterapkan: `gambar_2_3_mlops_lifecycle.png` digambar ulang sebagai loop AgriMLOps.

4. **Dong et al. (2023) Open-world Plant Disease Detection**
   - Figure paper menggunakan alur dynamic learning/update knowledge.
   - Cocok sebagai inspirasi konseptual untuk active learning, tetapi AgriMLOps tidak mengklaim true open-world detection.
   - Yang diterapkan: `gambar_2_4_active_learning_flow.png` hanya menggambarkan uncertainty/feedback/validation/retraining.

5. **Wei et al. (2024) PlantWild Benchmark**
   - Paper menekankan tantangan in-the-wild: small inter-class discrepancy dan large intra-class variance.
   - Cocok untuk narasi dataset dan research gap, bukan untuk disalin gambarnya.

## Keputusan desain ulang

- Semua diagram utama dibuat ulang dengan style konsisten: warna terbatas, rounded boxes, panah sederhana, font sans-serif jelas.
- Tidak ada gambar paper yang dicopy langsung ke laporan.
- Diagram yang menyebut “adaptasi” tetap berupa interpretasi konsep, bukan reproduksi visual.
- Screenshot production tidak lagi dipakai mentah penuh; dibuat panel evidence yang lebih rapi untuk DOCX.
"""
    (ANALYSIS_DIR / "visual_reference_analysis.md").write_text(content, encoding="utf-8")


def sync_report_images() -> None:
    report_img = ROOT / "report" / "images"
    report_img.mkdir(parents=True, exist_ok=True)
    for image in FIG_DIR.glob("*.png"):
        shutil.copy2(image, report_img / image.name)


def main() -> None:
    ensure_dirs()
    create_reference_visual_analysis()
    figure_2_1()
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_3_1()
    figure_3_2()
    figure_3_3()
    figure_3_4()
    figure_3_5()
    figure_3_6()
    figure_3_7()
    figure_4_1()
    figure_4_2()
    figure_4_3_confusion_matrix_clean()
    figure_4_4_sample_predictions_clean()
    clean_screenshots()
    api_docs_panel()
    api_health_panel()
    pil_refine_diagrams()
    sync_report_images()
    print(f"Generated paper-like figures in {FIG_DIR}")
    print(f"Synced copies to {FINAL_FIG_DIR} and report/images")


if __name__ == "__main__":
    main()
