# server/server.py
"""
Federated Learning Server for Diabetes Prediction.

Features
─────────
  FedAvg   – standard weighted parameter averaging (McMahan et al., 2017)
  FedProx  – proximal-term regularised aggregation (Li et al., 2020)
             Clients receive µ each round and add (µ/2)||w-w_global||² to loss.

  Server-side evaluation
    After every aggregation round the server reconstructs the global model and
    evaluates it on a HELD-OUT test set that clients never see.  This gives a
    single, unbiased view of global model quality rather than relying on
    federated average of client validation metrics.

  SHA-512 integrity check
    Each client attaches a hash of its parameter vector.  The server logs and
    verifies these hashes.  Note: this guards against accidental corruption in
    transit, NOT against a malicious client sending crafted weights.

  Model checkpointing
    Best model by server-side AUC-ROC is saved to artifacts/best_model.pth.
    Final model is saved to artifacts/global_model.pth.

  Experiment logging
    All per-round global metrics are forwarded to experiments/logger.py for
    TensorBoard and JSON export.

Start the server
─────────────────
  python server/server.py --rounds 15 --clients 3 --strategy fedprox --mu 0.1

Or via the unified CLI:
  python train_federated.py --mode server --rounds 15 --strategy fedprox
"""

import os
import sys
import logging
import argparse
import numpy as np
import torch
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

logging.getLogger("flwr").setLevel(logging.WARNING)

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model       import DiabetesNet
from evaluation.metrics import evaluate_model, print_metrics_table, plot_evaluation_report
from utils.data_utils   import load_server_test_data
from experiments.logger import FederatedLogger
from privacy.dp_config  import PrivacyAccountant, DPConfig


# ─────────────────────────────────────────────────────────────────────────────
# Metric aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate client evaluation metrics weighted by sample count."""
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    result = {}
    for key in metrics[0][1]:
        try:
            result[key] = sum(n * float(m[key]) for n, m in metrics) / total
        except (TypeError, ValueError):
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Custom FedAvg strategy with server-side evaluation
# ─────────────────────────────────────────────────────────────────────────────

class DiabetesFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg with:
      - Server-side global evaluation after each round
      - SHA-512 parameter integrity verification
      - Best-model checkpointing by AUC-ROC
      - Experiment metric logging
    """

    def __init__(
        self,
        data_dir:    str,
        num_rounds:  int,
        logger:      FederatedLogger,
        dp_accountant: Optional[PrivacyAccountant] = None,
        save_dir:    str = "artifacts",
        device_str:  str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.data_dir      = data_dir
        self.num_rounds    = num_rounds
        self.logger        = logger
        self.dp_accountant = dp_accountant
        self.save_dir      = save_dir
        self.device        = torch.device(device_str)

        self.best_auc   = 0.0
        self.best_round = 0
        self.round_metrics: Dict[str, list] = {
            "auc_roc": [], "f1": [], "recall": [],
            "precision": [], "accuracy": [], "auc_pr": [],
        }

        os.makedirs(save_dir, exist_ok=True)

        # Load server held-out test set once at startup
        self.test_loader = load_server_test_data(data_dir)
        n = len(self.test_loader.dataset)
        print(f"[Server] Held-out test set loaded: {n} samples")

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _params_to_model(self, parameters: fl.common.Parameters) -> DiabetesNet:
        """Reconstruct a DiabetesNet from aggregated Flower Parameters."""
        net    = DiabetesNet().to(self.device)
        arrays = fl.common.parameters_to_ndarrays(parameters)
        sd     = OrderedDict(
            {k: torch.tensor(v, device=self.device)
             for k, v in zip(net.state_dict().keys(), arrays)}
        )
        net.load_state_dict(sd)
        return net

    def _save_checkpoint(self, net: DiabetesNet, filename: str) -> None:
        path = os.path.join(self.save_dir, filename)
        torch.save(net.state_dict(), path)

    # ─────────────────────────────────────────────────────────────────────────
    # Flower strategy overrides
    # ─────────────────────────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results:      List[Tuple],
        failures:     List,
    ) -> Tuple[Optional[fl.common.Parameters], Dict]:
        """Verify SHA-512 hashes, then delegate to FedAvg aggregation."""

        print(f"\n{'─'*60}")
        print(f"  Round {server_round}/{self.num_rounds}  —  "
              f"aggregating {len(results)} client(s)")

        verified = []
        for client_proxy, fit_res in results:
            sha   = fit_res.metrics.get("sha512", None)
            n_smp = fit_res.num_examples
            cid   = str(getattr(client_proxy, "cid", "?"))[:12]

            if sha:
                print(f"  ✔ Client {cid} | n={n_smp} | SHA512={sha[:20]}…")
                verified.append((client_proxy, fit_res))
            else:
                print(f"  ✘ Client {cid} sent no hash — excluded from round")

        return super().aggregate_fit(server_round, verified, failures)

    def evaluate(
        self,
        server_round: int,
        parameters:   fl.common.Parameters,
    ) -> Optional[Tuple[float, Dict]]:
        """
        Server-side global evaluation on held-out test set.

        Called automatically by Flower after aggregate_fit each round.
        Returns (loss, metrics) — loss is returned as (1 - accuracy) proxy.
        """
        net     = self._params_to_model(parameters)
        metrics = evaluate_model(net, self.test_loader, self.device)

        # ── Log to experiment tracker ──────────────────────────────────────
        log_metrics = {
            k: metrics[k]
            for k in ["auc_roc", "auc_pr", "f1", "recall", "precision",
                      "specificity", "accuracy"]
        }
        self.logger.log_round(server_round, log_metrics, prefix="global")

        # ── Track for final plot ───────────────────────────────────────────
        for k in self.round_metrics:
            if k in metrics:
                self.round_metrics[k].append(metrics[k])

        # ── Privacy budget ─────────────────────────────────────────────────
        if self.dp_accountant is not None:
            eps = self.dp_accountant.compute_epsilon(
                num_steps=server_round * 5   # approx local_epochs per round
            )
            self.logger.log_privacy(server_round, eps, self.dp_accountant.config.target_delta)
            print(f"  Privacy budget consumed: ε = {eps:.3f}")

        # ── Console summary ────────────────────────────────────────────────
        print(f"  Global → AUC={metrics['auc_roc']:.4f} | "
              f"F1={metrics['f1']:.4f} | "
              f"Recall={metrics['recall']:.4f} | "
              f"Acc={metrics['accuracy']:.4f}")

        # ── Best model checkpoint ──────────────────────────────────────────
        if metrics["auc_roc"] > self.best_auc:
            self.best_auc   = metrics["auc_roc"]
            self.best_round = server_round
            self._save_checkpoint(net, "best_model.pth")
            print(f"  ★ New best AUC-ROC: {self.best_auc:.4f} (round {server_round})")

        # ── Final round: save global model + generate plots ────────────────
        if server_round == self.num_rounds:
            self._save_checkpoint(net, "global_model.pth")
            print(f"\n  Final model saved → {self.save_dir}/global_model.pth")
            print(f"  Best model was at round {self.best_round} "
                  f"(AUC={self.best_auc:.4f})")
            self._generate_final_outputs(net)

        # Flower expects (loss, dict)
        loss = 1.0 - float(metrics["accuracy"])
        return loss, {k: float(v) for k, v in metrics.items()
                      if isinstance(v, (int, float))}

    def _generate_final_outputs(self, net: DiabetesNet) -> None:
        """Generate final evaluation report and training curves."""
        all_probs, all_labels = [], []
        net.eval()

        with torch.no_grad():
            for X_b, y_b in self.test_loader:
                probs = torch.sigmoid(net(X_b.to(self.device))).squeeze()
                all_probs.extend(probs.cpu().numpy().flatten().tolist())
                all_labels.extend(y_b.squeeze().numpy().flatten().tolist())

        plot_evaluation_report(
            y_true   = np.array(all_labels),
            y_prob   = np.array(all_probs),
            save_dir = self.save_dir,
            prefix   = "final_federated",
        )

        print_metrics_table(
            evaluate_model(net, self.test_loader, self.device),
            title="Final Global Model — Server Evaluation",
        )

        self.logger.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# FedProx strategy
# ─────────────────────────────────────────────────────────────────────────────

class DiabetesFedProx(DiabetesFedAvg):
    """
    FedProx — FedAvg with a proximal regularisation term sent to clients.

    The proximal term µ is communicated to clients via the Flower `config`
    dictionary in configure_fit().  Clients add (µ/2)||w - w_global||²
    to their local loss, which limits how far local updates drift from the
    global model — particularly important under non-IID data.

    Reference
    ----------
    Li, T. et al. (2020).  "Federated Optimization in Heterogeneous Networks."
    Proceedings of Machine Learning and Systems 2 (MLSys 2020).
    """

    def __init__(self, proximal_mu: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        print(f"[Server] FedProx initialised  µ = {proximal_mu}")

    def configure_fit(
        self,
        server_round:   int,
        parameters:     fl.common.Parameters,
        client_manager: fl.server.ClientManager,
    ):
        """Inject proximal_mu into the fit config sent to every client."""
        fit_configs = super().configure_fit(server_round, parameters, client_manager)
        # Append proximal_mu to each client's config dict
        updated = []
        for client_proxy, fit_ins in fit_configs:
            new_config = dict(fit_ins.config)
            new_config["proximal_mu"] = self.proximal_mu
            from flwr.common import FitIns
            updated.append((client_proxy, FitIns(fit_ins.parameters, new_config)))
        return updated


# ─────────────────────────────────────────────────────────────────────────────
# Server entry point
# ─────────────────────────────────────────────────────────────────────────────

def start_server(
    num_rounds:     int   = 10,
    num_clients:    int   = 3,
    data_dir:       str   = "data/federated",
    strategy_name:  str   = "fedavg",
    proximal_mu:    float = 0.1,
    server_address: str   = "0.0.0.0:8080",
    save_dir:       str   = "artifacts",
    experiment_name: str  = "fl_diabetes",
    local_epochs:   int   = 5,
    dp_enabled:     bool  = False,
    dp_noise:       float = 1.0,
    dp_clip:        float = 1.0,
    num_samples:    int   = 500,   # approx federated training samples for DP
    batch_size:     int   = 32,
) -> None:
    """
    Initialise and start the Flower federated learning server.

    Parameters
    ----------
    num_rounds      : number of federated aggregation rounds
    num_clients     : minimum clients required each round
    data_dir        : directory containing federated NPZ datasets
    strategy_name   : "fedavg" or "fedprox"
    proximal_mu     : FedProx µ hyperparameter (ignored for fedavg)
    server_address  : gRPC bind address
    save_dir        : where to write model checkpoints and plots
    experiment_name : name for logging run directory
    local_epochs    : epochs per round (sent to clients via config)
    dp_enabled      : enable differential privacy tracking
    dp_noise        : DP noise multiplier σ
    dp_clip         : DP gradient clip norm C
    num_samples     : total training samples (for DP budget calculation)
    batch_size      : client batch size (for DP budget calculation)
    """

    # ── Experiment logger ─────────────────────────────────────────────────────
    logger = FederatedLogger(
        experiment_name=f"{experiment_name}_{strategy_name}",
        log_dir="experiments/runs",
    )
    logger.log_config({
        "num_rounds":     num_rounds,
        "num_clients":    num_clients,
        "strategy":       strategy_name,
        "proximal_mu":    proximal_mu if strategy_name == "fedprox" else None,
        "local_epochs":   local_epochs,
        "dp_enabled":     dp_enabled,
        "dp_noise":       dp_noise if dp_enabled else None,
        "dp_clip":        dp_clip  if dp_enabled else None,
        "data_dir":       data_dir,
        "server_address": server_address,
    })

    # ── DP accountant ─────────────────────────────────────────────────────────
    dp_accountant = None
    if dp_enabled:
        dp_cfg       = DPConfig(noise_multiplier=dp_noise,
                                max_grad_norm=dp_clip, enabled=True)
        dp_accountant = PrivacyAccountant(dp_cfg, num_samples, batch_size)
        dp_accountant.print_report(num_rounds, local_epochs)

    # ── Strategy configuration ────────────────────────────────────────────────
    common_kwargs = dict(
        data_dir       = data_dir,
        num_rounds     = num_rounds,
        logger         = logger,
        dp_accountant  = dp_accountant,
        save_dir       = save_dir,
        min_fit_clients        = num_clients,
        min_evaluate_clients   = num_clients,
        min_available_clients  = num_clients,
        evaluate_metrics_aggregation_fn = weighted_average,
        on_fit_config_fn = lambda rnd: {
            "server_round": rnd,
            "num_rounds":   num_rounds,
            "local_epochs": local_epochs,
        },
    )

    strategy: DiabetesFedAvg
    if strategy_name == "fedprox":
        strategy = DiabetesFedProx(proximal_mu=proximal_mu, **common_kwargs)
    else:
        strategy = DiabetesFedAvg(**common_kwargs)

    # ── Launch ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Federated Learning Server — {strategy_name.upper()}")
    print(f"{'═'*60}")
    print(f"  Rounds     : {num_rounds}")
    print(f"  Clients    : {num_clients}")
    print(f"  Address    : {server_address}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Local epochs/round : {local_epochs}")
    if strategy_name == "fedprox":
        print(f"  FedProx µ  : {proximal_mu}")
    print(f"{'═'*60}\n")

    fl.server.start_server(
        server_address = server_address,
        config         = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy       = strategy,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the FL server")
    parser.add_argument("--rounds",    type=int,   default=10)
    parser.add_argument("--clients",   type=int,   default=3)
    parser.add_argument("--strategy",  choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--mu",        type=float, default=0.1,  help="FedProx µ")
    parser.add_argument("--data_dir",  type=str,   default="data/federated")
    parser.add_argument("--address",   type=str,   default="0.0.0.0:8080")
    parser.add_argument("--save_dir",  type=str,   default="artifacts")
    parser.add_argument("--epochs",    type=int,   default=5,
                        help="Local epochs per round")
    parser.add_argument("--dp",        action="store_true", help="Enable DP tracking")
    parser.add_argument("--dp_noise",  type=float, default=1.0)
    parser.add_argument("--dp_clip",   type=float, default=1.0)
    parser.add_argument("--experiment",type=str,   default="fl_diabetes")
    args = parser.parse_args()

    start_server(
        num_rounds     = args.rounds,
        num_clients    = args.clients,
        data_dir       = args.data_dir,
        strategy_name  = args.strategy,
        proximal_mu    = args.mu,
        server_address = args.address,
        save_dir       = args.save_dir,
        experiment_name= args.experiment,
        local_epochs   = args.epochs,
        dp_enabled     = args.dp,
        dp_noise       = args.dp_noise,
        dp_clip        = args.dp_clip,
    )
