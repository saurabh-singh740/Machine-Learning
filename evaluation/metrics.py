# evaluation/metrics.py
"""
Comprehensive clinical evaluation metrics for the diabetes prediction model.

Why accuracy alone is insufficient in healthcare ML
────────────────────────────────────────────────────
The Pima dataset has ~35 % positive cases.  A trivial classifier that always
predicts "No Diabetes" achieves ~65 % accuracy while missing *every* diabetic
patient — completely useless clinically.

Metrics implemented
───────────────────
  AUC-ROC      – threshold-independent discrimination (primary metric)
  AUC-PR       – better than AUC-ROC for imbalanced data
  F1 Score     – harmonic mean of precision and recall
  Recall       – sensitivity / true-positive rate (most critical in healthcare)
  Precision    – positive predictive value
  Specificity  – true-negative rate
  Confusion matrix – breakdown of TP / TN / FP / FN

Threshold selection
────────────────────
  Default 0.5 is not optimal for imbalanced medical data.
  We implement Youden's J statistic (= TPR − FPR) to find the operating
  point that maximises the sum of sensitivity and specificity.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Threshold selection
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """
    Find the decision threshold that maximises Youden's J statistic.

    Youden's J = TPR − FPR  (ranges from 0 to 1, higher is better).

    For a clinical deployment you might prefer a lower threshold to bias
    toward higher sensitivity (catching more true diabetics at the cost of
    more false alarms).  Youden's J gives a principled, data-driven default.

    Returns
    -------
    (optimal_threshold, youden_j_score)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores  = tpr - fpr
    best_idx  = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true:       np.ndarray,
    y_pred_binary: np.ndarray,
    y_prob:       np.ndarray,
    threshold:    float = 0.5,
) -> Dict[str, float]:
    """
    Compute full clinical metric suite from predictions.

    Parameters
    ----------
    y_true        : ground-truth binary labels  (0 or 1)
    y_pred_binary : hard predictions at `threshold`
    y_prob        : predicted probabilities  [0.0, 1.0]
    threshold     : decision threshold used to produce y_pred_binary

    Returns
    -------
    Dictionary containing all metrics.
    """
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        # ── Primary metrics ───────────────────────────────────────────────
        "auc_roc":     float(roc_auc_score(y_true, y_prob)),
        "auc_pr":      float(average_precision_score(y_true, y_prob)),
        "f1":          float(f1_score(y_true, y_pred_binary, zero_division=0)),

        # ── Per-class metrics ─────────────────────────────────────────────
        "recall":      float(recall_score(y_true, y_pred_binary, zero_division=0)),   # sensitivity
        "precision":   float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "specificity": specificity,

        # ── Overall accuracy ──────────────────────────────────────────────
        "accuracy":    float((tp + tn) / (tp + tn + fp + fn)),

        # ── Confusion matrix cells ────────────────────────────────────────
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),

        # ── Meta ──────────────────────────────────────────────────────────
        "threshold":   float(threshold),
        "support":     float(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model evaluation on a DataLoader
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    threshold:  Optional[float] = None,
) -> Dict[str, float]:
    """
    Run inference on `dataloader` and return all clinical metrics.

    Parameters
    ----------
    model      : trained PyTorch model (outputs raw logit or probability)
    dataloader : evaluation DataLoader
    device     : torch.device
    threshold  : if None, automatically determined via Youden's J

    Returns
    -------
    Full metrics dictionary from compute_metrics().
    """
    model.eval()
    all_probs:  list = []
    all_labels: list = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch).squeeze()
        probs   = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(np.atleast_1d(probs).tolist())
        all_labels.extend(y_batch.squeeze().numpy().flatten().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    return compute_metrics(y_true, y_pred, y_prob, threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Console display
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics: Dict[str, float], title: str = "Evaluation") -> None:
    """Pretty-print metrics dictionary to stdout."""
    w = 54
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")
    print(f"  AUC-ROC      : {metrics.get('auc_roc',    0):.4f}  ← primary metric")
    print(f"  AUC-PR       : {metrics.get('auc_pr',     0):.4f}")
    print(f"  F1 Score     : {metrics.get('f1',         0):.4f}")
    print(f"  Recall       : {metrics.get('recall',     0):.4f}  (Sensitivity / TPR)")
    print(f"  Precision    : {metrics.get('precision',  0):.4f}")
    print(f"  Specificity  : {metrics.get('specificity',0):.4f}  (TNR)")
    print(f"  Accuracy     : {metrics.get('accuracy',   0):.4f}")
    print(f"  Threshold    : {metrics.get('threshold',  0):.4f}")
    print(f"{'─'*w}")
    tp = int(metrics.get("tp", 0))
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    print(f"  TP={tp:<5} FP={fp:<5}  (predicted positive)")
    print(f"  FN={fn:<5} TN={tn:<5}  (predicted negative)")
    print(f"{'═'*w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_evaluation_report(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    save_dir:  str,
    prefix:    str = "evaluation",
    threshold: Optional[float] = None,
) -> None:
    """
    Generate and save a 4-panel evaluation figure:
      1. ROC Curve          – with AUC and optimal operating point
      2. Precision-Recall   – with average precision score
      3. Confusion Matrix   – with TP/TN/FP/FN counts
      4. Probability Dist   – class-conditional score histograms
    """
    os.makedirs(save_dir, exist_ok=True)

    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    # ── 1. ROC Curve ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"ROC Curve (AUC = {auc_roc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    # mark Youden's J optimal point
    opt_idx = int(np.argmax(tpr - fpr))
    ax1.scatter(fpr[opt_idx], tpr[opt_idx], color="red", s=90, zorder=5,
                label=f"Optimal Threshold = {thresholds_roc[opt_idx]:.3f}")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])

    # ── 2. Precision-Recall Curve ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = float(y_true.mean())
    ax2.plot(recall_vals, precision_vals, color="darkorange", lw=2,
             label=f"PR Curve (AP = {ap:.3f})")
    ax2.axhline(baseline, color="grey", linestyle="--", lw=1,
                label=f"No-Skill baseline ({baseline:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])

    # ── 3. Confusion Matrix ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cm   = confusion_matrix(y_true, y_pred)
    im   = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax3)
    classes    = ["No Diabetes", "Diabetes"]
    tick_marks = np.arange(2)
    ax3.set_xticks(tick_marks); ax3.set_xticklabels(classes, fontsize=10)
    ax3.set_yticks(tick_marks); ax3.set_yticklabels(classes, fontsize=10)
    thresh_cm = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]),
                     ha="center", va="center", fontsize=15, fontweight="bold",
                     color="white" if cm[i, j] > thresh_cm else "black")
    ax3.set_ylabel("True Label")
    ax3.set_xlabel("Predicted Label")
    ax3.set_title(f"Confusion Matrix  (threshold = {threshold:.3f})")

    # ── 4. Probability Distribution ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 30)
    ax4.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, color="steelblue",
             label="No Diabetes", density=True)
    ax4.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, color="salmon",
             label="Diabetes", density=True)
    ax4.axvline(threshold, color="red", linestyle="--", lw=2,
                label=f"Threshold = {threshold:.3f}")
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Density")
    ax4.set_title("Score Distribution by Class")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "Diabetes Prediction — Evaluation Report",
        fontsize=14, fontweight="bold", y=1.01,
    )

    save_path = os.path.join(save_dir, f"{prefix}_report.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluation] Report saved → {save_path}")


def plot_fl_accuracy_curve(
    round_metrics: Dict[str, list],
    save_dir: str,
    prefix: str = "federated",
) -> None:
    """
    Plot global metric progression over federated rounds.

    Parameters
    ----------
    round_metrics : dict mapping metric name → list of per-round values
    save_dir      : directory to save the PNG
    prefix        : filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)

    metrics_to_plot = {k: v for k, v in round_metrics.items() if len(v) > 0}
    n = len(metrics_to_plot)
    if n == 0:
        return

    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).flatten()

    for ax, (name, values) in zip(axes, metrics_to_plot.items()):
        rounds = list(range(1, len(values) + 1))
        ax.plot(rounds, values, marker="o", linewidth=2, markersize=5)
        ax.set_xlabel("Federated Round")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Federated Learning — Global Metrics per Round",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluation] Training curves saved → {save_path}")
