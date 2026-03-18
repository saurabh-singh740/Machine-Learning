# run_project.py
"""
Single-terminal launcher for the Federated Learning Diabetes system.

Runs the entire pipeline inside the CURRENT terminal (VS Code or any shell):
  1. Prepare federated datasets      (blocking, skipped if already done)
  2. Start FL server                 (background subprocess)
  3. Start N clients                 (background subprocesses, one per client)
  4. Stream all logs here            (colour-prefixed so you can tell them apart)
  5. Wait for training to finish
  6. Run prediction interactively    (python inference.py)

Usage
──────
  python run_project.py                     # 3 clients, 10 rounds, FedAvg
  python run_project.py --clients 2         # 2 clients
  python run_project.py --rounds 15         # 15 FL rounds
  python run_project.py --strategy fedprox  # FedProx aggregation
  python run_project.py --alpha 0.1         # stronger non-IID data split
  python run_project.py --skip_prepare      # skip data prep if already done
  python run_project.py --skip_predict      # skip prediction step
  python run_project.py --server_delay 6    # give server more time to start
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON       = sys.executable          # same interpreter running this script

# ANSI colour codes — VS Code terminal supports these on all platforms
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_COLOURS = {
    "Launcher": "\033[97m",   # white
    "Server":   "\033[94m",   # blue
    "Client 0": "\033[92m",   # green
    "Client 1": "\033[93m",   # yellow
    "Client 2": "\033[95m",   # magenta
    "Client 3": "\033[96m",   # cyan
    "Predict":  "\033[96m",   # cyan
}


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _colour(label: str) -> str:
    return _COLOURS.get(label, "\033[97m")


def launcher_log(msg: str) -> None:
    """Print a message from the launcher itself."""
    ts    = time.strftime("%H:%M:%S")
    color = _colour("Launcher")
    print(f"{ts} {color}{_BOLD}[Launcher]{_RESET} {msg}", flush=True)


def _print_line(label: str, line: str) -> None:
    """Print one output line from a subprocess, with coloured label prefix."""
    ts    = time.strftime("%H:%M:%S")
    color = _colour(label)
    print(f"{ts} {color}[{label}]{_RESET} {line}", flush=True)


def divider(title: str = "") -> None:
    line = "─" * 60
    if title:
        pad = max(0, (60 - len(title) - 2) // 2)
        print(f"\n{'─' * pad} {title} {'─' * (60 - pad - len(title) - 2)}\n",
              flush=True)
    else:
        print(line, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_env() -> dict:
    """
    Return an environment dict with PYTHONUNBUFFERED=1.

    This forces Python subprocesses to flush stdout line-by-line instead of
    holding output in an internal buffer.  Without this, logs from the server
    and clients would appear in large bursts rather than in real time.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _stream_worker(proc: subprocess.Popen, label: str) -> None:
    """
    Background thread: read lines from `proc.stdout` and print them with
    a coloured `[label]` prefix so you can distinguish server vs clients.

    stdout and stderr are merged into a single pipe (stderr=STDOUT), so
    this one thread captures everything from the subprocess.
    """
    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:                           # skip blank lines
                _print_line(label, line)
    except ValueError:
        pass    # pipe closed — normal at process exit


def start_background(cmd: list, label: str) -> tuple:
    """
    Launch `cmd` as a background subprocess and start a reader thread.

    Returns (Popen, Thread).

    Key flags
    ──────────
    stdout=PIPE      – capture output so we can prefix it
    stderr=STDOUT    – merge stderr into stdout (one pipe, one thread)
    text=False       – binary mode; we decode manually (handles encoding errors)
    bufsize=0        – no buffering on our end (PYTHONUNBUFFERED handles child)
    """
    proc = subprocess.Popen(
        cmd,
        stdout  = subprocess.PIPE,
        stderr  = subprocess.STDOUT,   # merge stderr → single stream
        cwd     = PROJECT_ROOT,
        env     = _make_env(),
        bufsize = 0,                   # read bytes as soon as available
    )
    thread = threading.Thread(
        target  = _stream_worker,
        args    = (proc, label),
        daemon  = True,                # thread dies when main thread exits
        name    = f"reader-{label}",
    )
    thread.start()
    return proc, thread


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def step_prepare(num_clients: int, alpha: float) -> None:
    """
    Prepare federated datasets.  Runs in the current terminal (blocking).
    Skips automatically if all expected files already exist.
    """
    divider("Step 1 — Prepare Federated Datasets")

    data_dir     = PROJECT_ROOT / "data" / "federated"
    client_files = [data_dir / f"client_{i}.npz" for i in range(num_clients)]
    server_test  = data_dir / "server_test.npz"

    if server_test.exists() and all(f.exists() for f in client_files):
        launcher_log("Datasets already exist — skipping preparation.")
        return

    launcher_log(f"Generating datasets for {num_clients} clients (α={alpha}) …")

    result = subprocess.run(
        [
            PYTHON, "train_federated.py",
            "--mode",        "prepare",
            "--num_clients", str(num_clients),
            "--alpha",       str(alpha),
        ],
        cwd = PROJECT_ROOT,
        env = _make_env(),
    )

    if result.returncode != 0:
        launcher_log("ERROR: Data preparation failed. Aborting.")
        sys.exit(1)

    launcher_log("Datasets ready.")


def step_server(
    num_clients:  int,
    num_rounds:   int,
    strategy:     str,
    mu:           float,
    local_epochs: int,
) -> tuple:
    """Start the FL server in the background and return (Popen, Thread)."""
    divider("Step 2 — Start FL Server")

    cmd = [
        PYTHON, "train_federated.py",
        "--mode",     "server",
        "--rounds",   str(num_rounds),
        "--clients",  str(num_clients),
        "--strategy", strategy,
        "--mu",       str(mu),
        "--epochs",   str(local_epochs),
    ]

    launcher_log(
        f"Starting server  [{strategy.upper()}  {num_rounds} rounds  "
        f"{num_clients} clients]"
    )
    proc, thread = start_background(cmd, "Server")
    launcher_log("Server process started (PID {})".format(proc.pid))
    return proc, thread


def step_clients(
    num_clients: int,
    num_rounds:  int,
    dp:          bool,
    dp_noise:    float,
    dp_clip:     float,
    client_delay: float,
) -> list:
    """
    Start all clients in the background.

    Returns list of (Popen, Thread) tuples.
    """
    divider("Step 3 — Start Clients")

    procs = []
    for cid in range(num_clients):
        cmd = [
            PYTHON, "train_federated.py",
            "--mode",      "client",
            "--client_id", str(cid),
            "--rounds",    str(num_rounds),
        ]
        if dp:
            cmd += ["--dp", "--noise", str(dp_noise), "--clip", str(dp_clip)]

        label = f"Client {cid}"
        launcher_log(
            f"Starting {label}  [DP={'ON  σ={} C={}'.format(dp_noise, dp_clip) if dp else 'OFF'}]"
        )
        proc, thread = start_background(cmd, label)
        launcher_log(f"{label} process started (PID {proc.pid})")
        procs.append((proc, thread))

        if cid < num_clients - 1:
            time.sleep(client_delay)

    return procs


def step_wait(server_proc: subprocess.Popen,
              client_procs: list,
              all_threads:  list) -> bool:
    """
    Block until every client AND the server have exited.

    Returns True if all exited with code 0, False if any failed.
    """
    divider("Waiting for training to complete")
    launcher_log("Training in progress — logs are streaming above …")
    launcher_log("Press Ctrl+C to abort.")

    # Wait for clients first (they finish when all rounds are done)
    for proc, _ in client_procs:
        proc.wait()

    # Then wait for server (it may take a moment longer to finalise)
    server_proc.wait()

    # Let reader threads drain any remaining output
    for t in all_threads:
        t.join(timeout=3.0)

    # Check exit codes
    codes   = [p.returncode for p, _ in client_procs]
    codes  += [server_proc.returncode]
    success = all(c == 0 for c in codes)

    if success:
        launcher_log("All processes finished successfully.")
    else:
        launcher_log(f"WARNING: some processes exited with non-zero codes: {codes}")

    return success


def step_predict() -> None:
    """
    Run the inference script interactively in the current terminal.

    Uses subprocess.run() without any piping so stdin/stdout/stderr are
    fully connected to the VS Code terminal — the user can type normally.
    """
    divider("Step 5 — Run Prediction")
    launcher_log("Launching prediction script …")
    launcher_log("Enter patient data when prompted.\n")

    subprocess.run(
        [PYTHON, "inference.py"],
        cwd = PROJECT_ROOT,
        # No stdout/stderr redirection → inherits this terminal's handles
    )


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description = "Single-terminal FL launcher",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = __doc__,
    )

    # Training
    p.add_argument("--clients",      type=int,   default=3,
                   help="Number of federated clients  (default: 3)")
    p.add_argument("--rounds",       type=int,   default=10,
                   help="FL aggregation rounds  (default: 10)")
    p.add_argument("--strategy",     choices=["fedavg", "fedprox"],
                   default="fedavg",
                   help="Aggregation strategy  (default: fedavg)")
    p.add_argument("--mu",           type=float, default=0.1,
                   help="FedProx µ  (default: 0.1)")
    p.add_argument("--epochs",       type=int,   default=5,
                   help="Local epochs per round  (default: 5)")

    # Data
    p.add_argument("--alpha",        type=float, default=0.5,
                   help="Dirichlet α  (default: 0.5)")
    p.add_argument("--skip_prepare", action="store_true",
                   help="Skip dataset preparation")

    # DP
    p.add_argument("--dp",           action="store_true",
                   help="Enable Differential Privacy on clients")
    p.add_argument("--dp_noise",     type=float, default=1.0,
                   help="DP noise multiplier σ  (default: 1.0)")
    p.add_argument("--dp_clip",      type=float, default=1.0,
                   help="DP gradient clip norm C  (default: 1.0)")

    # Timing
    p.add_argument("--server_delay", type=float, default=5.0,
                   help="Seconds to wait after server start before "
                        "launching clients  (default: 5)")
    p.add_argument("--client_delay", type=float, default=1.5,
                   help="Seconds between consecutive client starts  "
                        "(default: 1.5)")

    # Misc
    p.add_argument("--skip_predict", action="store_true",
                   help="Skip the prediction step after training")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    divider()
    print(f"""
  {_BOLD}Federated Learning — Type 2 Diabetes Prediction{_RESET}
  Strategy : {args.strategy.upper()}
  Clients  : {args.clients}
  Rounds   : {args.rounds}
  Epochs   : {args.epochs} per round
  Alpha    : {args.alpha}  (Dirichlet non-IID)
  DP       : {'ON  (σ={} C={})'.format(args.dp_noise, args.dp_clip) if args.dp else 'OFF'}
    """, flush=True)
    divider()

    # ── Collect all processes for Ctrl+C cleanup ──────────────────────────────
    all_procs   = []
    all_threads = []

    def _cleanup(sig=None, _frame=None):
        launcher_log("Interrupted — terminating all subprocesses …")
        for proc in all_procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Prepare data
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_prepare:
        step_prepare(args.clients, args.alpha)
    else:
        launcher_log("--skip_prepare: skipping dataset generation.")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Server
    # ─────────────────────────────────────────────────────────────────────────
    server_proc, server_thread = step_server(
        num_clients  = args.clients,
        num_rounds   = args.rounds,
        strategy     = args.strategy,
        mu           = args.mu,
        local_epochs = args.epochs,
    )
    all_procs.append(server_proc)
    all_threads.append(server_thread)

    launcher_log(
        f"Waiting {args.server_delay}s for server to initialise "
        f"before starting clients …"
    )
    time.sleep(args.server_delay)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Clients
    # ─────────────────────────────────────────────────────────────────────────
    client_procs = step_clients(
        num_clients  = args.clients,
        num_rounds   = args.rounds,
        dp           = args.dp,
        dp_noise     = args.dp_noise,
        dp_clip      = args.dp_clip,
        client_delay = args.client_delay,
    )
    for proc, thread in client_procs:
        all_procs.append(proc)
        all_threads.append(thread)

    launcher_log(f"All {args.clients} client(s) running.")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Wait for training
    # ─────────────────────────────────────────────────────────────────────────
    step_wait(server_proc, client_procs, all_threads)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Prediction
    # ─────────────────────────────────────────────────────────────────────────
    divider()
    launcher_log("Federated training complete.")
    launcher_log("Artifacts saved to:  artifacts/")
    launcher_log("Metrics saved to  :  experiments/runs/")

    if not args.skip_predict:
        step_predict()
    else:
        launcher_log("--skip_predict: skipping inference step.")

    divider("Done")


if __name__ == "__main__":
    main()
