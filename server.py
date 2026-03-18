# server.py
"""
Simple wrapper — starts the FL server.

Usage:
    python server.py
"""

import subprocess
import sys

NUM_ROUNDS  = 10
NUM_CLIENTS = 3
STRATEGY    = "fedavg"

print(f"\nStarting FL Server  [strategy={STRATEGY}  rounds={NUM_ROUNDS}  clients={NUM_CLIENTS}]")
print("Waiting for clients to connect ...\n")

subprocess.run([
    sys.executable, "train_federated.py",
    "--mode",     "server",
    "--rounds",   str(NUM_ROUNDS),
    "--clients",  str(NUM_CLIENTS),
    "--strategy", STRATEGY,
])
