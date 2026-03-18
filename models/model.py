# models/model.py
"""
Upgraded DiabetesNet Neural Network.

Key improvements over the baseline:
  - BatchNorm1d: stabilises training across heterogeneous federated clients
    whose local data distributions can differ significantly (non-IID).
  - Dropout: prevents overfitting on small per-client datasets.
  - Kaiming He weight initialisation: optimal for ReLU networks.
  - Raw logit output (no sigmoid): use BCEWithLogitsLoss during training
    for numerically stable gradient computation.
  - predict_proba(): sigmoid applied only at inference time.

Architecture:  8 → [64, BN, ReLU, Drop(0.3)] → [32, BN, ReLU, Drop(0.3)]
                 → [16, BN, ReLU, Drop(0.15)] → 1 (logit)
"""

import torch
import torch.nn as nn
from typing import Optional


class DiabetesNet(nn.Module):
    """
    Feedforward neural network for binary diabetes classification.

    Parameters
    ----------
    input_dim    : number of input features (default 8 for Pima dataset)
    dropout_rate : base dropout probability; halved for the last hidden block
    """

    def __init__(self, input_dim: int = 8, dropout_rate: float = 0.3):
        super(DiabetesNet, self).__init__()

        self.network = nn.Sequential(
            # ── Block 1 ──────────────────────────────────────────────────────
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # ── Block 2 ──────────────────────────────────────────────────────
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # ── Block 3 (lighter dropout near output) ────────────────────────
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),

            # ── Output — raw logit (sigmoid applied externally) ───────────────
            nn.Linear(16, 1),
        )

        self._initialize_weights()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_weights(self) -> None:
        """Kaiming He initialisation for ReLU layers; constant 1/0 for BN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit — pair with BCEWithLogitsLoss during training."""
        return self.network(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return predicted probability in [0, 1] via sigmoid.
        Use during inference only (not training).
        """
        self.eval()
        return torch.sigmoid(self.forward(x))

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total = self.count_parameters()
        return (
            f"DiabetesNet(\n"
            f"  8 → [64·BN·ReLU·Drop] → [32·BN·ReLU·Drop] "
            f"→ [16·BN·ReLU·Drop] → 1\n"
            f"  Trainable parameters: {total:,}\n"
            f")"
        )
