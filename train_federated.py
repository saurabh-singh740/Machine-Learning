# train_federated.py
"""
Unified CLI entry point for the Federated Learning Diabetes system.

Modes
──────
  prepare      – generate federated datasets (run ONCE before training)
  server       – start the aggregation server
  client       – start a federated client
  centralized  – run the centralised baseline for comparison

Quick-start (3 terminals)
──────────────────────────
  # Terminal 1 — prepare data (run once)
  python train_federated.py --mode prepare --num_clients 3 --alpha 0.5

  # Terminal 2 — start server
  python train_federated.py --mode server --rounds 15 --clients 3

  # Terminal 3 — start client 0
  python train_federated.py --mode client --client_id 0

  # Terminal 4 — start client 1
  python train_federated.py --mode client --client_id 1

  # Terminal 5 — start client 2
  python train_federated.py --mode client --client_id 2

  # Optional: centralised baseline
  python train_federated.py --mode centralized
"""

import argparse
import sys
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_federated.py",
        description="Federated Learning for Type 2 Diabetes Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=["prepare", "server", "client", "centralized"],
        default="server",
        help="Execution mode (default: server)",
    )

    # ── Data / partitioning ───────────────────────────────────────────────────
    data = parser.add_argument_group("Data")
    data.add_argument("--num_clients", type=int, default=3,
                      help="Number of federated clients")
    data.add_argument("--alpha",       type=float, default=0.5,
                      help="Dirichlet α for non-IID partitioning "
                           "(lower = more heterogeneous)")
    data.add_argument("--data_dir",    type=str, default="data/federated",
                      help="Directory for client NPZ files and server test set")
    data.add_argument("--visualize",   action="store_true",
                      help="(prepare mode) Save partition visualisation PNG")

    # ── Server ────────────────────────────────────────────────────────────────
    srv = parser.add_argument_group("Server")
    srv.add_argument("--rounds",    type=int,   default=10,
                     help="Number of FL aggregation rounds")
    srv.add_argument("--strategy",  choices=["fedavg", "fedprox"],
                     default="fedavg",
                     help="Aggregation strategy")
    srv.add_argument("--mu",        type=float, default=0.1,
                     help="FedProx proximal term µ (only for fedprox)")
    srv.add_argument("--address",   type=str,   default="0.0.0.0:8080",
                     help="Server bind address")
    srv.add_argument("--save_dir",  type=str,   default="artifacts",
                     help="Directory to save model checkpoints and plots")
    srv.add_argument("--epochs",    type=int,   default=5,
                     help="Local training epochs per round")
    srv.add_argument("--experiment",type=str,   default="fl_diabetes",
                     help="Experiment name for logging")

    # ── Client ────────────────────────────────────────────────────────────────
    cli = parser.add_argument_group("Client")
    cli.add_argument("--client_id",     type=int, default=0,
                     help="Client ID  (must match a data/federated/client_N.npz file)")
    cli.add_argument("--server_address",type=str, default="127.0.0.1:8080",
                     help="Server address to connect to")

    # ── Differential Privacy ──────────────────────────────────────────────────
    dp = parser.add_argument_group("Differential Privacy")
    dp.add_argument("--dp",    action="store_true",
                    help="Enable DP gradient clipping + noise on clients")
    dp.add_argument("--noise", type=float, default=1.0,
                    help="DP noise multiplier σ (higher = more private)")
    dp.add_argument("--clip",  type=float, default=1.0,
                    help="DP gradient clip norm C")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    if args.mode == "prepare":
        from data_partitioning.partitioner import (
            create_federated_datasets,
            visualize_partitions,
        )

        print("\n" + "=" * 60)
        print("  Step 1 / 3 — Preparing Federated Datasets")
        print("=" * 60)

        create_federated_datasets(
            output_dir  = args.data_dir,
            num_clients = args.num_clients,
            alpha       = args.alpha,
        )

        if args.visualize:
            visualize_partitions(
                num_clients = args.num_clients,
                save_path   = "artifacts/partition_visualization.png",
            )

        print("\nDatasets ready.  Next steps:")
        print(f"  Terminal A: python train_federated.py --mode server "
              f"--rounds {args.rounds} --clients {args.num_clients}")
        for i in range(args.num_clients):
            print(f"  Terminal {chr(66+i)}: python train_federated.py "
                  f"--mode client --client_id {i}")

    # ─────────────────────────────────────────────────────────────────────────
    elif args.mode == "server":
        from server.server import start_server

        start_server(
            num_rounds      = args.rounds,
            num_clients     = args.num_clients,
            data_dir        = args.data_dir,
            strategy_name   = args.strategy,
            proximal_mu     = args.mu,
            server_address  = args.address,
            save_dir        = args.save_dir,
            experiment_name = args.experiment,
            local_epochs    = args.epochs,
            dp_enabled      = args.dp,
            dp_noise        = args.noise,
            dp_clip         = args.clip,
        )

    # ─────────────────────────────────────────────────────────────────────────
    elif args.mode == "client":
        from client.client import start_client

        start_client(
            client_id        = args.client_id,
            data_dir         = args.data_dir,
            server_address   = args.server_address,
            num_rounds       = args.rounds,
            dp_enabled       = args.dp,
            noise_multiplier = args.noise,
            max_grad_norm    = args.clip,
        )

    # ─────────────────────────────────────────────────────────────────────────
    elif args.mode == "centralized":
        from centralized import run_centralized

        run_centralized(save_dir=args.save_dir)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
