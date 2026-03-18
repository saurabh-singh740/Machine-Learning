# experiments/logger.py
"""
Experiment tracking for federated learning runs.

Supports:
  - TensorBoard  (via torch.utils.tensorboard.SummaryWriter)
  - JSON file    (always enabled; readable without any viewer)
  - Console      (structured stdout output)

Usage
──────
  logger = FederatedLogger(experiment_name="fl_fedprox_alpha05")
  logger.log_config({"num_rounds": 15, "strategy": "fedprox", ...})

  # Inside the FL loop
  logger.log_round(round_num=3, metrics={"auc_roc": 0.81, "f1": 0.72},
                   prefix="global")
  logger.log_client_metrics(round_num=3, client_id=0,
                            metrics={"recall": 0.75})
  logger.log_privacy(round_num=3, epsilon=4.2, delta=1e-5)

  logger.finalize()   # saves curves, closes TensorBoard writer

View TensorBoard:
  tensorboard --logdir experiments/runs
"""

import os
import json
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Optional TensorBoard import (graceful fallback) ───────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Logger class
# ─────────────────────────────────────────────────────────────────────────────

class FederatedLogger:
    """
    Multi-backend experiment logger for federated learning.

    Parameters
    ----------
    experiment_name  : human-readable name for this run
    log_dir          : root directory; a timestamped sub-dir is created per run
    use_tensorboard  : enable TensorBoard writer if the library is available
    """

    def __init__(
        self,
        experiment_name: str  = "fl_diabetes",
        log_dir:         str  = "experiments/runs",
        use_tensorboard: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.start_time      = time.time()

        # Create a unique run directory
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Internal metric storage
        # Structure: { "prefix/metric_name": [{"round": int, "value": float}, …] }
        self.history: Dict[str, List[dict]] = defaultdict(list)
        self.config:  Dict = {}

        # JSON output path
        self.json_path = os.path.join(self.run_dir, "metrics.json")

        # TensorBoard
        self.writer = None
        if use_tensorboard and _TB_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.run_dir)
            print(f"[Logger] TensorBoard → {self.run_dir}")
            print(f"         View with : tensorboard --logdir {log_dir}")
        elif use_tensorboard and not _TB_AVAILABLE:
            print("[Logger] TensorBoard not installed — using JSON logging only.")

        print(f"[Logger] Run directory: {self.run_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # Configuration logging
    # ─────────────────────────────────────────────────────────────────────────

    def log_config(self, config: Dict) -> None:
        """
        Save experiment hyperparameters.

        Call once before training starts.  Writes config.json and
        logs hyperparameters to TensorBoard if available.
        """
        self.config = config
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        if self.writer:
            # TensorBoard hparams panel
            str_config = {k: str(v) for k, v in config.items()}
            self.writer.add_hparams(str_config, metric_dict={})

        print("[Logger] Config saved →", config_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-round metric logging
    # ─────────────────────────────────────────────────────────────────────────

    def log_round(
        self,
        round_num: int,
        metrics:   Dict[str, float],
        prefix:    str = "global",
    ) -> None:
        """
        Record metrics for a single federated round.

        Parameters
        ----------
        round_num : federated round index (1-based)
        metrics   : dict of metric_name → value
        prefix    : namespace prefix, e.g. "global", "client_0", "privacy"
        """
        for name, value in metrics.items():
            key = f"{prefix}/{name}"
            self.history[key].append({"round": round_num, "value": float(value)})
            if self.writer:
                self.writer.add_scalar(key, float(value), global_step=round_num)

        self._flush_json()

    def log_client_metrics(
        self,
        round_num: int,
        client_id: int,
        metrics:   Dict[str, float],
    ) -> None:
        """Log per-client evaluation metrics."""
        self.log_round(round_num, metrics, prefix=f"client_{client_id}")

    def log_privacy(
        self,
        round_num: int,
        epsilon:   float,
        delta:     float,
    ) -> None:
        """Log differential privacy budget consumed at this round."""
        self.log_round(
            round_num,
            {"epsilon": epsilon, "delta": delta},
            prefix="privacy",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # JSON persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_json(self) -> None:
        """Write current state to metrics.json (called after every log_round)."""
        payload = {
            "experiment":       self.experiment_name,
            "run_dir":          self.run_dir,
            "config":           self.config,
            "elapsed_seconds":  round(time.time() - self.start_time, 1),
            "metrics":          dict(self.history),
        }
        with open(self.json_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    # ─────────────────────────────────────────────────────────────────────────
    # Plotting
    # ─────────────────────────────────────────────────────────────────────────

    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Generate training curve plots from all logged "global/*" metrics.

        Saves a multi-panel PNG to the run directory.
        """
        global_entries = {
            k: v for k, v in self.history.items()
            if k.startswith("global/") and len(v) > 0
        }

        if not global_entries:
            print("[Logger] No global metrics to plot yet.")
            return

        if save_path is None:
            save_path = os.path.join(self.run_dir, "training_curves.png")

        n    = len(global_entries)
        cols = 2
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes      = np.array(axes).flatten()

        for ax, (key, data) in zip(axes, global_entries.items()):
            rounds = [d["round"] for d in data]
            values = [d["value"] for d in data]
            metric_label = key.split("/")[-1].replace("_", " ").title()

            ax.plot(rounds, values, marker="o", linewidth=2, markersize=5,
                    color="steelblue")
            ax.set_xlabel("Federated Round")
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            ax.grid(True, alpha=0.3)

            # Annotate final value
            if values:
                ax.annotate(
                    f"{values[-1]:.4f}",
                    xy=(rounds[-1], values[-1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="steelblue",
                )

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle(
            f"Federated Learning — {self.experiment_name}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Logger] Training curves saved → {save_path}")

    def plot_convergence_comparison(
        self,
        centralized_metrics: Optional[Dict[str, float]] = None,
        save_path:           Optional[str]              = None,
    ) -> None:
        """
        Plot federated AUC-ROC vs centralized reference line (if provided).

        Useful for quantifying the "cost of federation".
        """
        key = "global/auc_roc"
        if key not in self.history or not self.history[key]:
            print("[Logger] No global/auc_roc data to compare.")
            return

        if save_path is None:
            save_path = os.path.join(self.run_dir, "convergence_comparison.png")

        data   = self.history[key]
        rounds = [d["round"] for d in data]
        fl_auc = [d["value"] for d in data]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(rounds, fl_auc, marker="o", linewidth=2, color="steelblue",
                label="Federated (Global AUC-ROC)")

        if centralized_metrics and "auc_roc" in centralized_metrics:
            c_auc = centralized_metrics["auc_roc"]
            ax.axhline(c_auc, color="red", linestyle="--", lw=2,
                       label=f"Centralized (AUC-ROC = {c_auc:.4f})")
            # shaded gap
            ax.fill_between(rounds, fl_auc, c_auc,
                            alpha=0.08, color="red", label="Federation cost")

        ax.set_xlabel("Federated Round")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("Federated vs Centralised Convergence")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Logger] Convergence comparison saved → {save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Finalisation
    # ─────────────────────────────────────────────────────────────────────────

    def finalize(
        self,
        centralized_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Close all writers, generate final plots, and print a summary.

        Call once after all federated rounds are complete.
        """
        elapsed = time.time() - self.start_time
        print(f"\n[Logger] Experiment complete in {elapsed:.1f}s")

        self.plot_training_curves()
        self.plot_convergence_comparison(centralized_metrics)
        self._flush_json()

        if self.writer:
            self.writer.close()

        print(f"[Logger] All outputs saved to: {self.run_dir}")
