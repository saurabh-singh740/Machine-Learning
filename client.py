# client.py
"""
Simple wrapper — starts one FL client.

Usage:
    python client.py
    → Enter client id (0, 1, 2): 0
"""

import subprocess
import sys

NUM_CLIENTS = 3
NUM_ROUNDS  = 10

# ── Ask for client ID ─────────────────────────────────────────────────────────
while True:
    raw = input(f"Enter client id (0 – {NUM_CLIENTS - 1}): ").strip()
    if raw.isdigit() and 0 <= int(raw) < NUM_CLIENTS:
        client_id = int(raw)
        break
    print(f"  Invalid input. Please enter a number between 0 and {NUM_CLIENTS - 1}.")

print(f"\nStarting Client {client_id}  [rounds={NUM_ROUNDS}]\n")

subprocess.run([
    sys.executable, "train_federated.py",
    "--mode",      "client",
    "--client_id", str(client_id),
    "--rounds",    str(NUM_ROUNDS),
])
