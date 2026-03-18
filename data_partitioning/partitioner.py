# data_partitioning/partitioner.py
"""
Non-IID Federated Data Partitioner using the Dirichlet Distribution.

Why Non-IID partitioning?
──────────────────────────
Real hospital datasets are NOT uniformly distributed.  Different hospitals
serve different patient demographics — a paediatric clinic has few elderly
patients; a rural clinic may have higher rates of metabolic disease; an
endocrinology specialist sees predominantly diabetic patients.

FedAvg was shown to converge significantly slower (or diverge) under strong
non-IID conditions (Li et al., 2020).  Simulating non-IID partitions is
therefore essential for a realistic federated learning evaluation.

Dirichlet(α) concentration parameter
──────────────────────────────────────
  α → ∞     :  IID     — all clients see the same class distribution
  α = 1.0   :  moderate heterogeneity
  α = 0.5   :  high heterogeneity  ← recommended default for realistic sim
  α → 0     :  extreme non-IID — each client receives only one class

Usage
──────
  # Generate datasets for 3 clients with high heterogeneity:
  python data_partitioning/partitioner.py --num_clients 3 --alpha 0.5

  # Also visualise how the partition looks across alphas:
  python data_partitioning/partitioner.py --num_clients 3 --alpha 0.5 --visualize
"""

import os
import sys
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import load_raw_data, compute_pos_weight, DATA_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Core partitioning logic
# ─────────────────────────────────────────────────────────────────────────────

def partition_data_dirichlet(
    X:            np.ndarray,
    y:            np.ndarray,
    num_clients:  int,
    alpha:        float = 0.5,
    random_state: int   = 42,
    min_samples:  int   = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Assign samples to clients via per-class Dirichlet draws.

    Algorithm
    ----------
    For each class c:
      1. Collect the indices of all samples belonging to class c.
      2. Draw a proportion vector p ~ Dirichlet(α · 1_K) over K clients.
      3. Assign floor(p_k · |class_c|) samples to client k.
      4. Give the rounding remainder to the last client.

    This produces heterogeneous class distributions while guaranteeing
    every client receives at least `min_samples` samples overall.

    Parameters
    ----------
    X, y         : full feature matrix and label vector (unscaled)
    num_clients  : number of federated clients
    alpha        : Dirichlet concentration (lower → more non-IID)
    random_state : reproducibility seed
    min_samples  : minimum samples any client must have

    Returns
    -------
    List of (X_client, y_client) tuples in original (unscaled) feature space.
    """
    rng     = np.random.default_rng(random_state)
    classes = np.unique(y)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)

        # Draw proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Convert to integer counts
        counts = np.floor(proportions * len(cls_idx)).astype(int)

        # Remainder goes to the last client
        remainder = len(cls_idx) - counts.sum()
        counts[-1] += remainder
        counts = np.maximum(counts, 0)

        # Assign
        ptr = 0
        for k, cnt in enumerate(counts):
            client_indices[k].extend(cls_idx[ptr: ptr + cnt].tolist())
            ptr += cnt

    # Shuffle within each client and build arrays
    partitions: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx_list in client_indices:
        arr = np.array(idx_list, dtype=int)
        if len(arr) < min_samples:
            # Fallback: give this client random samples to reach minimum
            extra = rng.choice(len(y), size=min_samples - len(arr), replace=False)
            arr   = np.concatenate([arr, extra])
        rng.shuffle(arr)
        partitions.append((X[arr], y[arr]))

    return partitions


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: load → split → scale → partition → save
# ─────────────────────────────────────────────────────────────────────────────

def create_federated_datasets(
    data_path:    str   = DATA_PATH,
    output_dir:   str   = "data/federated",
    num_clients:  int   = 3,
    alpha:        float = 0.5,
    server_test_size: float = 0.15,   # fraction reserved for server test set
    client_val_size:  float = 0.15,   # fraction of each client's data for val
    random_state: int   = 42,
) -> None:
    """
    End-to-end dataset generation for a federated simulation.

    Outputs
    -------
    data/federated/
      client_0.npz  …  client_{N-1}.npz   – per-client train/val splits
      server_test.npz                       – server held-out test set

    artifacts/
      scaler.pkl    – StandardScaler fit on the federated training pool

    Pipeline
    ---------
    1. Load diabetes.csv
    2. Reserve server test set (clients NEVER see this)
    3. Fit StandardScaler on remaining (federated) data — ONLY HERE
    4. Scale all splits using those statistics
    5. Apply Dirichlet partitioning across clients
    6. For each client: stratified train/val split → save NPZ
    7. Save server test NPZ and scaler
    """
    os.makedirs(output_dir, exist_ok=True)
    artifacts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts",
    )
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"\n{'═'*58}")
    print(f"  Federated Dataset Generation")
    print(f"{'═'*58}")
    print(f"  Clients    : {num_clients}")
    print(f"  Alpha      : {alpha}  (lower → more non-IID)")
    print(f"  Output dir : {output_dir}\n")

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    X, y = load_raw_data(data_path)
    print(f"  Loaded: {len(X)} samples | "
          f"{np.sum(y==0)} negative ({np.mean(y==0)*100:.1f}%) | "
          f"{np.sum(y==1)} positive ({np.mean(y)*100:.1f}%)")

    # ── Step 2: Reserve server test set ──────────────────────────────────────
    X_fed, X_srv_test, y_fed, y_srv_test = train_test_split(
        X, y,
        test_size=server_test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"\n  Server test set  : {len(X_srv_test)} samples (held-out, never shared)")
    print(f"  Federated pool   : {len(X_fed)} samples")

    # ── Step 3: Fit scaler on federated pool ONLY ─────────────────────────────
    scaler          = StandardScaler()
    X_fed_scaled    = scaler.fit_transform(X_fed)
    X_srv_test_sc   = scaler.transform(X_srv_test)   # use federated statistics

    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved     : {scaler_path}")

    # ── Step 4: Dirichlet partition of scaled federated data ──────────────────
    print(f"\n  Applying Dirichlet(α={alpha}) partitioning…")
    partitions = partition_data_dirichlet(
        X_fed_scaled, y_fed, num_clients, alpha, random_state
    )

    # ── Step 5: Per-client train/val split and save ───────────────────────────
    print(f"\n  {'Client':>8} {'Train':>8} {'Val':>8} {'Pos%':>8}")
    print(f"  {'─'*36}")

    for cid, (X_c, y_c) in enumerate(partitions):
        # Stratified split where possible
        try:
            X_c_tr, X_c_val, y_c_tr, y_c_val = train_test_split(
                X_c, y_c,
                test_size=client_val_size,
                random_state=random_state,
                stratify=y_c,
            )
        except ValueError:
            # If one class too small for stratified split, fall back to random
            X_c_tr, X_c_val, y_c_tr, y_c_val = train_test_split(
                X_c, y_c,
                test_size=client_val_size,
                random_state=random_state,
            )

        pos_pct = np.mean(y_c_tr) * 100
        print(f"  {cid:>8} {len(X_c_tr):>8} {len(X_c_val):>8} {pos_pct:>7.1f}%")

        save_path = os.path.join(output_dir, f"client_{cid}.npz")
        np.savez_compressed(
            save_path,
            X_train   = X_c_tr.astype(np.float32),
            y_train   = y_c_tr.astype(np.float32),
            X_val     = X_c_val.astype(np.float32),
            y_val     = y_c_val.astype(np.float32),
            client_id = np.array([cid]),
            alpha     = np.array([alpha]),
        )

    # ── Step 6: Save server test set ─────────────────────────────────────────
    srv_path = os.path.join(output_dir, "server_test.npz")
    np.savez_compressed(
        srv_path,
        X_test = X_srv_test_sc.astype(np.float32),
        y_test = y_srv_test.astype(np.float32),
    )
    print(f"\n  Server test NPZ  : {srv_path}")
    print(f"  Done. All datasets ready for training.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def visualize_partitions(
    data_path:   str         = DATA_PATH,
    num_clients: int         = 3,
    alphas:      List[float] = [10.0, 1.0, 0.5, 0.1],
    save_path:   str         = "artifacts/partition_visualization.png",
) -> None:
    """
    Compare class distribution across clients for several alpha values.

    Produces a bar-chart grid (one column per alpha) showing what fraction
    of each client's training data is diabetic (positive class).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X, y = load_raw_data(data_path)
    global_pos = float(np.mean(y))

    colors = ["steelblue", "darkorange", "green", "purple", "brown"]

    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 4),
                             sharey=True)

    for ax, alpha in zip(axes, alphas):
        parts = partition_data_dirichlet(X, y, num_clients, alpha)

        pos_ratios = [float(np.mean(yc)) for _, yc in parts]
        sizes      = [len(yc) for _, yc in parts]

        bars = ax.bar(
            range(num_clients), pos_ratios,
            color=colors[:num_clients], alpha=0.8,
        )
        ax.axhline(global_pos, color="red", linestyle="--", lw=1.5,
                   label=f"Global ({global_pos:.2f})")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Client ID")
        ax.set_title(f"α = {alpha}")
        ax.set_xticks(range(num_clients))
        ax.legend(fontsize=7)

        for bar, sz in zip(bars, sizes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={sz}", ha="center", va="bottom", fontsize=8,
            )

    axes[0].set_ylabel("Fraction Diabetic (Positive Class)")
    fig.suptitle(
        "Non-IID Partitioning: Class Distribution per Client\n"
        "(lower α → greater heterogeneity)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Partitioner] Visualisation saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate federated datasets with non-IID partitioning"
    )
    parser.add_argument("--num_clients", type=int, default=3,
                        help="Number of federated clients")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration (0.5 = high heterogeneity)")
    parser.add_argument("--output_dir", type=str, default="data/federated",
                        help="Directory to save client NPZ files")
    parser.add_argument("--visualize", action="store_true",
                        help="Also generate partition visualisation PNG")
    args = parser.parse_args()

    create_federated_datasets(
        num_clients=args.num_clients,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )

    if args.visualize:
        visualize_partitions(
            num_clients=args.num_clients,
            save_path="artifacts/partition_visualization.png",
        )
