# client/client.py
"""
Federated Learning Client for Diabetes Prediction.

Local training improvements over baseline
───────────────────────────────────────────
  Adam optimizer           – faster convergence than SGD for tabular data
  Cosine LR scheduler      – smooth learning rate decay across FL rounds
  BCEWithLogitsLoss + pos_weight – class-imbalance aware loss
  FedProx proximal term    – (µ/2)||w-w_global||² added to local loss;
                             prevents excessive client drift under non-IID data
  Differential Privacy     – gradient clipping + Gaussian noise injection
  SHA-512 integrity hash   – attached to every model update sent to server
  Full local evaluation    – AUC-ROC, F1, Recall, Precision on local val set

Start a client
───────────────
  python client/client.py --client_id 0 --data_dir data/federated

Or via the unified CLI:
  python train_federated.py --mode client --client_id 0

With DP:
  python client/client.py --client_id 0 --dp --noise 1.0 --clip 1.0
"""

import os
import sys
import argparse
import hashlib
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model       import DiabetesNet
from utils.data_utils   import load_client_data
from evaluation.metrics import evaluate_model, print_metrics_table
from privacy.dp_config  import DPConfig, DPGradientClipper

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """Return model state-dict values as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a list of numpy arrays back into the model."""
    pairs = zip(net.state_dict().keys(), parameters)
    state = OrderedDict({k: torch.tensor(v) for k, v in pairs})
    net.load_state_dict(state, strict=True)


def compute_sha512(parameters: List[np.ndarray]) -> str:
    """
    Compute SHA-512 hash of the parameter list.

    Purpose: transport integrity — detects accidental corruption in the
    network layer.  Does NOT protect against a malicious client deliberately
    sending crafted weights (that requires SecAgg or cryptographic MACs).
    """
    raw  = pickle.dumps([p.tobytes() for p in parameters])
    return hashlib.sha512(raw).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_cosine_lr(
    server_round: int,
    num_rounds:   int,
    lr_max:       float = 1e-3,
    lr_min:       float = 1e-6,
) -> float:
    """
    Compute the cosine-annealed learning rate for the current FL round.

    Formula (mirrors CosineAnnealingLR):
        lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
    where t = server_round - 1  and  T = num_rounds - 1.

    Why we pre-compute LR instead of using CosineAnnealingLR.last_epoch
    ─────────────────────────────────────────────────────────────────────
    In FL, each call to fit() creates a FRESH optimizer.  Passing
    last_epoch > -1 to CosineAnnealingLR on a fresh optimizer raises:

        KeyError: "param 'initial_lr' is not specified in param_groups[0]
                   when resuming an optimizer"

    This happens because PyTorch's _LRScheduler.__init__ tries to replay
    the LR history by calling get_lr(), which expects 'initial_lr' in
    optimizer.param_groups — a key that is only written during the very
    first scheduler initialisation (last_epoch = -1 path).

    Solution: compute the round's LR analytically and inject it directly
    into Adam.  The scheduler is then initialised normally (last_epoch=-1)
    and used only for within-round stepping if desired.
    """
    t = max(0, server_round - 1)
    T = max(1, num_rounds - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t / T))


# ─────────────────────────────────────────────────────────────────────────────
# Local training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_local(
    net:           nn.Module,
    trainloader:   torch.utils.data.DataLoader,
    epochs:        int,
    pos_weight:    float,
    dp_config:     DPConfig,
    global_params: Optional[List[np.ndarray]] = None,
    proximal_mu:   float = 0.0,
    server_round:  int   = 1,
    num_rounds:    int   = 10,
) -> Dict[str, float]:
    """
    Train `net` on local data for `epochs` epochs.

    Design decisions
    ─────────────────
    Optimizer : Adam (lr=1e-3, weight_decay=1e-4)
      - Better than plain SGD for tabular healthcare data
      - Weight decay adds L2 regularisation
      - β₁, β₂ defaults (0.9, 0.999) — standard

    LR Schedule : cosine annealing across FL rounds
      - Round LR computed analytically via compute_cosine_lr()
        and injected directly into Adam (avoids PyTorch KeyError when
        last_epoch > -1 is passed to a freshly-created optimizer)
      - A secondary within-round CosineAnnealingLR(T_max=epochs) adds
        a gentle local warm-down over the local training epochs

    Loss : BCEWithLogitsLoss(pos_weight)
      - Numerically more stable than Sigmoid + BCELoss
      - pos_weight = neg/pos upweights the minority diabetic class

    FedProx : (µ/2) · ||w_local - w_global||²  added to loss
      - Only active when proximal_mu > 0 and global_params provided
      - Prevents local optima drifting far from the global model

    Differential Privacy :
      1. Clip aggregate gradient to ℓ₂-norm ≤ C  (via DPGradientClipper)
      2. Add Gaussian noise N(0, (σC/B)²) to each gradient tensor
         where σ = noise_multiplier, C = max_grad_norm, B = batch_size

    Parameters
    ──────────
    net           : model to train (modified in-place)
    trainloader   : local DataLoader
    epochs        : number of local epochs
    pos_weight    : class imbalance weight for loss function
    dp_config     : Differential Privacy configuration
    global_params : global model weights for FedProx proximal term
    proximal_mu   : FedProx µ  (0 = disabled)
    server_round  : current FL round number (used for LR scheduling)
    num_rounds    : total FL rounds (used for LR scheduling)

    Returns
    ───────
    Dict with "train_loss" and "learning_rate".
    """
    net.train()

    # ── Learning rate for this FL round (cosine-annealed globally) ────────────
    # Pre-compute the cosine LR analytically so we can pass it directly to Adam.
    # This avoids the PyTorch KeyError that occurs when last_epoch > -1 is
    # passed to CosineAnnealingLR on a freshly-created optimizer (no 'initial_lr'
    # key in param_groups yet).
    round_lr = compute_cosine_lr(server_round, num_rounds, lr_max=1e-3, lr_min=1e-6)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=round_lr,          # ← round-specific LR from cosine schedule
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    # Optional: a simple within-round linear warm-down over local epochs.
    # Initialised with last_epoch=-1 (default) → no 'initial_lr' bug.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = max(epochs, 1),   # decay over the local epochs of this round
        eta_min = round_lr * 0.1,   # floor = 10 % of round LR
        # last_epoch is NOT set → defaults to -1 → safe on a fresh optimizer
    )

    # Class-imbalance aware loss
    pw      = torch.tensor([pos_weight], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    # DP gradient clipper (only instantiated when DP is on)
    dp_clipper = DPGradientClipper(dp_config.max_grad_norm) if dp_config.enabled else None

    # Freeze a snapshot of the global weights for the FedProx proximal term
    global_tensors: Optional[List[torch.Tensor]] = None
    if proximal_mu > 0.0 and global_params is not None:
        global_tensors = [
            torch.tensor(p, device=DEVICE, requires_grad=False)
            for p in global_params
        ]

    cumulative_loss = 0.0

    for _epoch in range(epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in trainloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            batch_size = X_batch.size(0)

            optimizer.zero_grad()

            # ── Forward ──────────────────────────────────────────────────────
            logits = net(X_batch).squeeze()
            loss   = loss_fn(logits, y_batch.squeeze())

            # ── FedProx proximal term ─────────────────────────────────────────
            # (µ/2) · Σ ||w_k - w_global||²
            if global_tensors is not None and proximal_mu > 0.0:
                proximal = sum(
                    ((lp - gp) ** 2).sum()
                    for lp, gp in zip(net.parameters(), global_tensors)
                )
                loss = loss + (proximal_mu / 2.0) * proximal

            # ── Backward ─────────────────────────────────────────────────────
            loss.backward()

            # ── DP: clip gradients ─────────────────────────────────────────
            if dp_clipper is not None:
                dp_clipper.clip_and_measure(net)
                # Add Gaussian noise:  N(0, (σ · C / B)²)
                noise_std = (dp_config.noise_multiplier
                             * dp_config.max_grad_norm
                             / max(batch_size, 1))
                for param in net.parameters():
                    if param.grad is not None:
                        param.grad.data.add_(
                            torch.normal(
                                mean=0.0, std=noise_std,
                                size=param.grad.shape,
                                device=DEVICE,
                            )
                        )

            optimizer.step()
            epoch_loss += loss.item()

        cumulative_loss += epoch_loss / max(len(trainloader), 1)
        scheduler.step()   # step within-round scheduler after each local epoch

    avg_loss   = cumulative_loss / max(epochs, 1)
    # Report the round-level LR (the cosine value for this FL round),
    # not the within-round scheduler's final value.
    current_lr = round_lr

    return {"train_loss": avg_loss, "learning_rate": current_lr}


# ─────────────────────────────────────────────────────────────────────────────
# Flower NumPyClient
# ─────────────────────────────────────────────────────────────────────────────

class DiabetesClient(fl.client.NumPyClient):
    """
    Flower federated learning client for diabetes prediction.

    Interface methods
    ──────────────────
    get_parameters() : return current local model weights
    fit()            : receive global weights, train locally, return update
    evaluate()       : evaluate global weights on local validation set
    """

    def __init__(
        self,
        client_id:  int,
        data_dir:   str,
        dp_config:  DPConfig,
        num_rounds: int = 10,
    ) -> None:
        self.client_id  = client_id
        self.dp_config  = dp_config
        self.num_rounds = num_rounds

        # Load pre-partitioned client data (NPZ files — no access to global CSV)
        self.trainloader, self.valloader, self.pos_weight = load_client_data(
            client_id  = client_id,
            data_dir   = data_dir,
        )

        self.model = DiabetesNet().to(DEVICE)

        print(f"\n{'═'*55}")
        print(f"  Client {client_id} — ready")
        print(f"{'═'*55}")
        print(f"  Train samples  : {len(self.trainloader.dataset)}")
        print(f"  Val   samples  : {len(self.valloader.dataset)}")
        print(f"  Positive weight: {self.pos_weight:.3f}")
        print(f"  Device         : {DEVICE}")
        print(f"  Model params   : {self.model.count_parameters():,}")
        print(f"  DP enabled     : {dp_config.enabled}")
        if dp_config.enabled:
            print(f"    σ (noise)    : {dp_config.noise_multiplier}")
            print(f"    C (clip)     : {dp_config.max_grad_norm}")
        print(f"{'═'*55}\n")

    # ── Flower interface ──────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config:     Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        1. Load global model weights into local model
        2. Train locally for `local_epochs` epochs
        3. Compute SHA-512 hash of updated weights
        4. Return (updated_weights, num_samples, metrics_dict)
        """
        # Load global model
        set_parameters(self.model, parameters)

        # Read training config sent by server
        server_round  = int(config.get("server_round", 1))
        local_epochs  = int(config.get("local_epochs", 5))
        proximal_mu   = float(config.get("proximal_mu", 0.0))

        print(f"[Client {self.client_id}] Round {server_round} | "
              f"epochs={local_epochs} | µ={proximal_mu}")

        # Train
        stats = train_local(
            net           = self.model,
            trainloader   = self.trainloader,
            epochs        = local_epochs,
            pos_weight    = self.pos_weight,
            dp_config     = self.dp_config,
            global_params = parameters if proximal_mu > 0.0 else None,
            proximal_mu   = proximal_mu,
            server_round  = server_round,
            num_rounds    = self.num_rounds,
        )

        print(f"  → Loss: {stats['train_loss']:.4f} | "
              f"LR: {stats['learning_rate']:.6f}")

        updated_params = get_parameters(self.model)
        sha            = compute_sha512(updated_params)

        return updated_params, len(self.trainloader.dataset), {
            "sha512":        sha,
            "train_loss":    float(stats["train_loss"]),
            "learning_rate": float(stats["learning_rate"]),
            "client_id":     float(self.client_id),
        }

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config:     Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the received global model on the local validation set.

        Returns
        ───────
        loss       : 1 - accuracy proxy
        num_samples: number of validation samples
        metrics    : dict with AUC, F1, Recall, Precision, Accuracy
        """
        set_parameters(self.model, parameters)
        metrics = evaluate_model(self.model, self.valloader, DEVICE)

        print(f"  [Client {self.client_id}] Val → "
              f"AUC={metrics['auc_roc']:.4f} | "
              f"F1={metrics['f1']:.4f} | "
              f"Recall={metrics['recall']:.4f}")

        return (
            float(1.0 - metrics["accuracy"]),   # loss
            len(self.valloader.dataset),
            {
                "accuracy":  float(metrics["accuracy"]),
                "auc_roc":   float(metrics["auc_roc"]),
                "f1":        float(metrics["f1"]),
                "recall":    float(metrics["recall"]),
                "precision": float(metrics["precision"]),
                "client_id": float(self.client_id),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def start_client(
    client_id:       int,
    data_dir:        str   = "data/federated",
    server_address:  str   = "127.0.0.1:8080",
    num_rounds:      int   = 10,
    dp_enabled:      bool  = False,
    noise_multiplier: float = 1.0,
    max_grad_norm:   float = 1.0,
) -> None:
    """Connect to the FL server and start the federated training loop."""

    dp_config = DPConfig(
        noise_multiplier = noise_multiplier,
        max_grad_norm    = max_grad_norm,
        enabled          = dp_enabled,
    )

    client = DiabetesClient(
        client_id  = client_id,
        data_dir   = data_dir,
        dp_config  = dp_config,
        num_rounds = num_rounds,
    )

    fl.client.start_numpy_client(
        server_address = server_address,
        client         = client,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start an FL client")
    parser.add_argument("--client_id",  type=int,   required=True,
                        help="Client index (0, 1, 2, …)")
    parser.add_argument("--data_dir",   type=str,   default="data/federated")
    parser.add_argument("--server",     type=str,   default="127.0.0.1:8080")
    parser.add_argument("--rounds",     type=int,   default=10)
    parser.add_argument("--dp",         action="store_true",
                        help="Enable Differential Privacy gradient noise")
    parser.add_argument("--noise",      type=float, default=1.0,
                        help="DP noise multiplier σ")
    parser.add_argument("--clip",       type=float, default=1.0,
                        help="DP gradient clip norm C")
    args = parser.parse_args()

    start_client(
        client_id        = args.client_id,
        data_dir         = args.data_dir,
        server_address   = args.server,
        num_rounds       = args.rounds,
        dp_enabled       = args.dp,
        noise_multiplier = args.noise,
        max_grad_norm    = args.clip,
    )
