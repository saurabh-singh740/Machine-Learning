# centralized.py
"""
Centralised Learning Baseline.

Purpose
────────
Train on the full diabetes dataset with no privacy constraints to establish
an UPPER BOUND on achievable accuracy.  The gap between the centralized and
federated results quantifies the "cost of federation":

  federation_cost = centralized_AUC − federated_AUC

A small gap (< 2 %) indicates the FL system is working well.

Improvements over the original centralized.py
──────────────────────────────────────────────
  ✅ Data leakage fixed   – scaler fit on training data only
  ✅ Stratified splits    – preserve class distribution
  ✅ Upgraded model       – DiabetesNet with BatchNorm + Dropout
  ✅ Adam + Cosine LR     – better convergence
  ✅ Weighted loss        – handles class imbalance
  ✅ Full metric suite    – AUC-ROC, F1, Recall, Precision, Specificity
  ✅ Best-model save      – by validation AUC-ROC (not final epoch)
  ✅ Evaluation plots     – ROC, PR curve, confusion matrix, score distribution

Usage
──────
  python centralized.py
  python train_federated.py --mode centralized
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model       import DiabetesNet
from utils.data_utils   import prepare_global_splits, make_dataloader
from evaluation.metrics import (
    evaluate_model, print_metrics_table,
    plot_evaluation_report,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_centralized(
    epochs:       int   = 30,
    batch_size:   int   = 32,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    save_dir:     str   = "artifacts",
) -> dict:
    """
    Train and evaluate a centralised model.

    Parameters
    ──────────
    epochs       : number of training epochs
    batch_size   : mini-batch size
    lr           : initial learning rate (Adam)
    weight_decay : L2 regularisation coefficient
    save_dir     : directory to save model checkpoint and plots

    Returns
    ───────
    Dictionary of test-set evaluation metrics.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "═" * 58)
    print("  Centralised Baseline Training")
    print("═" * 58)

    # ── Data loading (scaler fit on train only — no leakage) ──────────────────
    X_train, X_val, X_test, y_train, y_val, y_test, pos_weight = (
        prepare_global_splits()
    )

    trainloader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    valloader   = make_dataloader(X_val,   y_val,   batch_size=batch_size, shuffle=False)
    testloader  = make_dataloader(X_test,  y_test,  batch_size=batch_size, shuffle=False)

    print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")
    print(f"  Positive class weight : {pos_weight:.3f}")
    print(f"  Device : {DEVICE}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DiabetesNet().to(DEVICE)
    print(model)

    # ── Optimiser and scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Class-imbalance aware loss ────────────────────────────────────────────
    pw      = torch.tensor([pos_weight], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc  = 0.0
    best_epoch    = 0
    best_ckpt     = os.path.join(save_dir, "centralized_best.pth")

    header = f"{'Epoch':>6}  {'Train Loss':>11}  {'Val AUC':>9}  {'Val F1':>7}  {'LR':>10}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for X_b, y_b in trainloader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b).squeeze()
            loss   = loss_fn(logits, y_b.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(trainloader), 1)

        # ── Validate ──────────────────────────────────────────────────────────
        val_m  = evaluate_model(model, valloader, DEVICE)
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"{epoch:>6}  {avg_loss:>11.4f}  "
            f"{val_m['auc_roc']:>9.4f}  "
            f"{val_m['f1']:>7.4f}  "
            f"{lr_now:>10.6f}"
        )

        # Save best model by validation AUC-ROC
        if val_m["auc_roc"] > best_val_auc:
            best_val_auc = val_m["auc_roc"]
            best_epoch   = epoch
            torch.save(model.state_dict(), best_ckpt)

        scheduler.step()

    # ── Load best checkpoint ──────────────────────────────────────────────────
    print(f"\n  Best validation AUC-ROC : {best_val_auc:.4f}  (epoch {best_epoch})")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    # ── Final test-set evaluation ─────────────────────────────────────────────
    test_metrics = evaluate_model(model, testloader, DEVICE)
    print_metrics_table(test_metrics, title="Centralised — Test Set Evaluation")

    # ── Evaluation report PNG ─────────────────────────────────────────────────
    all_probs: list = []
    all_labels: list = []
    model.eval()
    with torch.no_grad():
        for X_b, y_b in testloader:
            probs = torch.sigmoid(model(X_b.to(DEVICE))).squeeze().cpu().numpy()
            all_probs.extend(np.atleast_1d(probs).tolist())
            all_labels.extend(y_b.squeeze().numpy().flatten().tolist())

    plot_evaluation_report(
        y_true   = np.array(all_labels),
        y_prob   = np.array(all_probs),
        save_dir = save_dir,
        prefix   = "centralized",
    )

    return test_metrics


if __name__ == "__main__":
    run_centralized()
