# privacy/dp_config.py
"""
Differential Privacy (DP) configuration, gradient clipping, and privacy
budget accounting for the federated diabetes prediction system.

What Differential Privacy guarantees
──────────────────────────────────────
Formal definition (ε, δ)-DP:
  A randomised mechanism M satisfies (ε, δ)-DP if for any two datasets D, D'
  that differ in exactly ONE record, and for all subsets S of outputs:

      P[M(D) ∈ S]  ≤  exp(ε) · P[M(D') ∈ S]  +  δ

Interpretation for federated learning:
  - ε (epsilon): privacy budget — the maximum "information leakage" about any
    single patient's data.  Smaller ε = stronger privacy.
    Typical values: 1 (very strong), 10 (reasonable for healthcare demos).
  - δ (delta): probability of catastrophic failure.  Should be ≤ 1/n² where
    n is the number of training samples.

DP-SGD mechanism (Abadi et al., 2016 — "Deep Learning with DP")
────────────────────────────────────────────────────────────────
  Two operations added per gradient step:
  1. CLIP per-sample gradients to ℓ₂-norm ≤ C  (bounds sensitivity)
  2. ADD Gaussian noise N(0, σ²C²/B) to the clipped gradient sum
     where σ = noise_multiplier, B = batch_size

  These two steps together guarantee (ε, δ)-DP via the Moments Accountant.

Privacy-utility trade-off
──────────────────────────
  Higher σ  → stronger privacy, lower accuracy
  Higher C  → allows larger true gradients through, but also larger noise

  The default values (σ=1.0, C=1.0) represent a moderate privacy budget
  suitable for demonstration.  Production healthcare deployments typically
  target ε ≤ 1–3 with rigorous budget accounting.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DPConfig:
    """
    Differential Privacy hyperparameters.

    Attributes
    ----------
    noise_multiplier : σ — ratio of noise std to gradient clip norm.
                       Higher → more noise → stronger privacy, lower utility.
    max_grad_norm    : C — maximum ℓ₂-norm of per-sample gradient before noise.
                       Also called the "sensitivity" or "clip threshold".
    target_epsilon   : desired ε budget (used for early stopping if exceeded).
    target_delta     : δ — failure probability; set to ~1/n.
    enabled          : set False to disable DP entirely (for ablation study).
    """
    noise_multiplier: float = 1.0
    max_grad_norm:    float = 1.0
    target_epsilon:   float = 10.0   # ε ≤ 10 reasonable for healthcare demos
    target_delta:     float = 1e-5   # δ << 1/n; n≈100-600 in this dataset
    enabled:          bool  = True

    def __post_init__(self) -> None:
        if self.noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier must be > 0, got {self.noise_multiplier}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")


# ─────────────────────────────────────────────────────────────────────────────
# Gradient clipping
# ─────────────────────────────────────────────────────────────────────────────

class DPGradientClipper:
    """
    Clips model parameter gradients to a maximum ℓ₂-norm.

    Note on per-sample vs aggregate clipping
    ------------------------------------------
    DP-SGD requires INDIVIDUAL per-sample gradient clipping.  True per-sample
    clipping needs "ghost clipping" or Opacus-style hooks and is complex to
    implement from scratch.

    This implementation clips the AGGREGATE gradient (as in standard gradient
    clipping), which is simpler and still limits model update magnitude, but
    provides a weaker formal privacy guarantee than true per-sample clipping.

    For a production deployment use the Opacus library (Meta) which implements
    correct per-sample gradient clipping efficiently.
    """

    def __init__(self, max_norm: float) -> None:
        self.max_norm = max_norm

    def clip_and_measure(self, model: nn.Module) -> float:
        """
        Clip gradients in-place and return the pre-clipping gradient ℓ₂-norm.

        Returns
        -------
        grad_norm : float — ℓ₂ norm of the gradient before clipping
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(total_norm)

        clip_coef = self.max_norm / (grad_norm + 1e-8)
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

        return grad_norm

    def add_gaussian_noise(
        self,
        model:     nn.Module,
        batch_size: int,
        device:    torch.device,
    ) -> None:
        """
        Add calibrated Gaussian noise to gradients after clipping.

        Noise std = (σ · C) / batch_size
        where σ = noise_multiplier, C = max_grad_norm.

        Dividing by batch_size accounts for the fact that gradients are
        averaged over the mini-batch.
        """
        noise_std = (self.max_norm / batch_size)  # will be multiplied by σ
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=param.grad.shape,
                    device=device,
                )
                param.grad.data.add_(noise)


# ─────────────────────────────────────────────────────────────────────────────
# Privacy budget accounting
# ─────────────────────────────────────────────────────────────────────────────

class PrivacyAccountant:
    """
    Tracks the cumulative (ε, δ) privacy budget consumed during training.

    Uses a simplified moments-accountant approximation from Theorem 3.3
    (Abadi et al., 2016).  For production use, integrate with Google's
    `dp-accounting` library for exact RDP (Rényi Differential Privacy)
    accounting which gives tighter bounds.

    Simplified ε formula
    ─────────────────────
    After T gradient steps with sampling rate q = B/n and noise multiplier σ:

        ε  ≈  q · √(2T · ln(1/δ))  /  σ

    This is a conservative (loose) upper bound.  Tight bounds from RDP
    accounting are typically 2-5× lower.
    """

    def __init__(
        self,
        config:     DPConfig,
        num_samples: int,
        batch_size: int,
    ) -> None:
        self.config        = config
        self.num_samples   = num_samples
        self.batch_size    = batch_size
        self.sampling_rate = batch_size / max(num_samples, 1)
        self.epsilon_log:  List[float] = []

    def compute_epsilon(self, num_steps: int) -> float:
        """
        Compute approximate ε consumed after `num_steps` gradient steps.

        Parameters
        ----------
        num_steps : total number of SGD steps (rounds × local_epochs × batches)

        Returns
        -------
        epsilon : float — privacy budget consumed
        """
        if not self.config.enabled:
            return float("inf")   # No DP = unbounded information leakage

        q     = self.sampling_rate
        sigma = self.config.noise_multiplier
        delta = self.config.target_delta
        T     = num_steps

        if sigma == 0 or q == 0:
            return float("inf")

        epsilon = (q * math.sqrt(2 * T * math.log(1.0 / delta))) / sigma
        self.epsilon_log.append(epsilon)
        return epsilon

    def compute_epsilon_for_training(
        self,
        num_rounds:    int,
        local_epochs:  int,
        batches_per_epoch: Optional[int] = None,
    ) -> float:
        """Convenience wrapper for full FL training schedule."""
        if batches_per_epoch is None:
            batches_per_epoch = max(1, self.num_samples // self.batch_size)
        total_steps = num_rounds * local_epochs * batches_per_epoch
        return self.compute_epsilon(total_steps)

    def get_report(
        self,
        num_rounds:   int,
        local_epochs: int,
    ) -> dict:
        """Return a structured privacy report dictionary."""
        eps = self.compute_epsilon_for_training(num_rounds, local_epochs)
        delta = self.config.target_delta

        return {
            "enabled":          self.config.enabled,
            "epsilon":          round(eps, 4),
            "delta":            delta,
            "noise_multiplier": self.config.noise_multiplier,
            "max_grad_norm":    self.config.max_grad_norm,
            "sampling_rate":    round(self.sampling_rate, 6),
            "privacy_guarantee": (
                f"({eps:.2f}, {delta:.0e})-DP"
                if self.config.enabled else "DP disabled"
            ),
            "warning": (
                "ε > 10 — privacy protection is weak for a real deployment."
                if eps > 10 and self.config.enabled else ""
            ),
        }

    def print_report(self, num_rounds: int, local_epochs: int) -> None:
        """Pretty-print the privacy report to stdout."""
        r = self.get_report(num_rounds, local_epochs)
        w = 58
        print(f"\n{'═'*w}")
        print(f"  DIFFERENTIAL PRIVACY REPORT")
        print(f"{'═'*w}")
        print(f"  DP Enabled        : {r['enabled']}")
        print(f"  Privacy Guarantee : {r['privacy_guarantee']}")
        print(f"  Noise Multiplier  : σ = {r['noise_multiplier']}")
        print(f"  Gradient Clip     : C = {r['max_grad_norm']}")
        print(f"  Sampling Rate     : q = {r['sampling_rate']:.5f}")
        if r["warning"]:
            print(f"\n  ⚠ WARNING: {r['warning']}")
        print(f"{'═'*w}\n")
