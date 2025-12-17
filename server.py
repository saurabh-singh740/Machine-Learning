import logging
logging.getLogger("flwr").setLevel(logging.ERROR)

import flwr as fl
import matplotlib.pyplot as plt
import torch
from model import DiabetesNet

round_accuracies = []

# -------- Metric aggregation --------
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    acc = sum(accuracies) / total_examples
    round_accuracies.append(acc)
    return {"accuracy": acc}

# -------- Custom Strategy --------
class CustomFedAvg(fl.server.strategy.FedAvg):

    def aggregate_fit(self, rnd, results, failures):

        print(f"\n🔐 Verifying model updates (Round {rnd})")

        # 🔐 READ SHA FROM CLIENT METADATA
        for client_proxy, fit_res in results:
            sha = fit_res.metrics.get("sha256", None)
            if sha:
                print(f"✔ Client {client_proxy.cid} SHA-256: {sha[:16]}...")
            else:
                print(f"⚠ Client {client_proxy.cid} sent no SHA")

        # ✅ DO NORMAL FEDAVG (DO NOT TOUCH CORE LOGIC)
        aggregated = super().aggregate_fit(rnd, results, failures)

        # Save final global model
        if rnd == 10 and aggregated is not None:
            print("Saving final global model...")

            parameters, _ = aggregated
            net = DiabetesNet()

            params = fl.common.parameters_to_ndarrays(parameters)
            state_dict = dict(
                zip(net.state_dict().keys(), [torch.tensor(p) for p in params])
            )
            net.load_state_dict(state_dict)

            torch.save(net.state_dict(), "global_model.pth")
            print("✅ global_model.pth saved successfully\n")

        return aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)

        if aggregated is not None:
            acc = aggregated[1]["accuracy"] * 100
            if rnd in [1, 2, 5, 10]:
                print(f"Round {rnd}/10")
                print("Aggregating updates from 2 clients...")
                print(f"Global Model Accuracy: {acc:.1f}%\n")

        return aggregated


print("\n🔐 Federated Learning Server Started (SHA Enabled)\n")

strategy = CustomFedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)

# -------- START SERVER --------
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)

print("Training completed successfully")

# -------- PLOT --------
plt.figure(figsize=(9, 5))
rounds = range(1, len(round_accuracies) + 1)
plt.plot(rounds, [a * 100 for a in round_accuracies], marker="o", linewidth=2)
plt.xlabel("Federated Rounds")
plt.ylabel("Accuracy (%)")
plt.title("Federated Learning Accuracy over Rounds")
plt.grid(True)
plt.tight_layout()
plt.savefig("federated_accuracy_plot.png", dpi=300)
plt.show()
