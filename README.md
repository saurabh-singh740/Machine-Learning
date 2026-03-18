# Federated Learning Framework for Type 2 Diabetes Prediction

A **privacy-preserving machine learning system** that trains a diabetes prediction model across multiple simulated hospital clients without sharing raw patient data.

---

## Problem Statement

Traditional machine learning for healthcare requires centralising patient data from multiple hospitals onto a single server — raising serious **privacy, regulatory, and ethical concerns** (GDPR, HIPAA).

**Federated Learning (FL)** solves this by keeping data local. Each hospital trains on its own patients and only shares **model parameter updates**, never raw records. A central server aggregates these updates to build a global model.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING SYSTEM                        │
│                                                                     │
│  ┌──────────────┐    model weights     ┌──────────────────────┐    │
│  │  Hospital 0  │──────────────────────►                      │    │
│  │  (Client 0)  │◄─────────────────────│   Central Server     │    │
│  │  local data  │    global model      │   (Aggregator)       │    │
│  └──────────────┘                      │                      │    │
│                                        │  ┌────────────────┐  │    │
│  ┌──────────────┐    model weights     │  │   FedAvg /     │  │    │
│  │  Hospital 1  │──────────────────────►  │   FedProx      │  │    │
│  │  (Client 1)  │◄─────────────────────│  │  Aggregation   │  │    │
│  │  local data  │    global model      │  └────────────────┘  │    │
│  └──────────────┘                      │                      │    │
│                                        │  ┌────────────────┐  │    │
│  ┌──────────────┐    model weights     │  │  Server-side   │  │    │
│  │  Hospital 2  │──────────────────────►  │  Evaluation    │  │    │
│  │  (Client 2)  │◄─────────────────────│  │  (held-out)    │  │    │
│  │  local data  │    global model      │  └────────────────┘  │    │
│  └──────────────┘                      └──────────────────────┘    │
│                                                                     │
│   Each hospital keeps patient data LOCAL — only model updates       │
│   travel over the network (with SHA-512 integrity check).          │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Flow

```
Round 1 … N:
  Server  ──► Broadcast global model weights to all clients
  Clients ──► Load global weights
            ► Train locally (Adam, Cosine LR, weighted BCELoss)
            ► Optionally apply FedProx proximal term
            ► Optionally apply DP gradient clipping + noise
            ► Compute SHA-512 hash of updated weights
            ► Send updated weights + hash + metrics back to server
  Server  ──► Verify SHA-512 hashes
            ► Aggregate via Weighted FedAvg
            ► Evaluate global model on held-out server test set
            ► Log AUC-ROC, F1, Recall, Precision, Accuracy
            ► Save best checkpoint by AUC-ROC
```

---

## Dataset

**Pima Indians Diabetes Dataset** — 768 samples, 8 features, binary outcome.

| Feature                   | Description                                |
|---------------------------|--------------------------------------------|
| Pregnancies               | Number of pregnancies                      |
| Glucose                   | Plasma glucose concentration (mg/dL)       |
| BloodPressure             | Diastolic blood pressure (mm Hg)           |
| SkinThickness             | Triceps skin fold thickness (mm)           |
| Insulin                   | 2-hour serum insulin (µU/mL)               |
| BMI                       | Body mass index (kg/m²)                    |
| DiabetesPedigreeFunction  | Genetic diabetes risk score                |
| Age                       | Age in years                               |
| **Outcome**               | **0 = No Diabetes, 1 = Diabetes**          |

Class distribution: ~65% negative, ~35% positive (imbalanced — handled by `pos_weight` in loss).

---

## Project Structure

```
Federated-Learning-Healthcare-Diabetes/
│
├── models/
│   └── model.py              Upgraded DiabetesNet (BatchNorm, Dropout, He init)
│
├── utils/
│   └── data_utils.py         Data loading, preprocessing (leakage-free)
│
├── evaluation/
│   └── metrics.py            AUC-ROC, F1, Recall, Confusion Matrix, plots
│
├── data_partitioning/
│   └── partitioner.py        Dirichlet non-IID partitioner
│
├── privacy/
│   └── dp_config.py          DP gradient clipping, noise, budget accounting
│
├── experiments/
│   └── logger.py             TensorBoard + JSON experiment tracking
│
├── server/
│   └── server.py             FedAvg / FedProx + server-side evaluation
│
├── client/
│   └── client.py             FL client (Adam, Cosine LR, FedProx, DP)
│
├── data/
│   └── federated/            Per-client NPZ files + server test set
│
├── artifacts/                Model checkpoints, scalers, plots
├── experiments/runs/         TensorBoard logs and JSON metrics
│
├── train_federated.py        Unified CLI entry point
├── centralized.py            Centralised baseline for comparison
├── inference.py              Interactive + batch prediction
├── diabetes.csv              Dataset
└── requirements.txt
```

---

## Model Architecture

```
Input (8 features)
    │
    ▼
Linear(8 → 64)
BatchNorm1d(64)
ReLU
Dropout(0.30)
    │
    ▼
Linear(64 → 32)
BatchNorm1d(32)
ReLU
Dropout(0.30)
    │
    ▼
Linear(32 → 16)
BatchNorm1d(16)
ReLU
Dropout(0.15)
    │
    ▼
Linear(16 → 1)  ← raw logit
    │
    ▼  sigmoid applied at inference
Probability [0, 1]
```

**Total trainable parameters**: ~4,273

**Key design choices**:
- **BatchNorm**: stabilises training when client data distributions differ (non-IID)
- **Dropout**: prevents overfitting on small local datasets
- **Raw logit output**: paired with `BCEWithLogitsLoss` for numerical stability
- **Kaiming He initialisation**: optimal for ReLU networks

---

## Training Pipeline

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Prepare Federated Datasets (run once)

```bash
python train_federated.py --mode prepare --num_clients 3 --alpha 0.5
```

This generates:
- `data/federated/client_0.npz`, `client_1.npz`, `client_2.npz` — local train/val
- `data/federated/server_test.npz` — server held-out evaluation set
- `artifacts/scaler.pkl` — fitted StandardScaler

The `--alpha` parameter controls heterogeneity:
- `alpha = 10.0` → near-IID (uniform distribution)
- `alpha = 0.5`  → high heterogeneity (recommended)
- `alpha = 0.1`  → extreme non-IID (one class per client)

### Step 3 — Start Training (3 separate terminals)

**Terminal 1 — Server:**
```bash
python train_federated.py --mode server --rounds 15 --clients 3 --strategy fedavg
```

**Terminal 2 — Client 0:**
```bash
python train_federated.py --mode client --client_id 0
```

**Terminal 3 — Client 1:**
```bash
python train_federated.py --mode client --client_id 1
```

**Terminal 4 — Client 2:**
```bash
python train_federated.py --mode client --client_id 2
```

### Step 4 — View Results

```bash
# TensorBoard
tensorboard --logdir experiments/runs

# Evaluation plots are saved automatically to artifacts/
```

---

## Aggregation Strategies

### FedAvg (default)
Standard federated averaging (McMahan et al., 2017).
Each client's update is weighted by its number of training samples.

```bash
python train_federated.py --mode server --strategy fedavg
```

### FedProx
Adds a proximal regularisation term to each client's local objective:

```
Local loss = F_i(w) + (µ/2) · ||w - w_global||²
```

This prevents client models from drifting too far from the global model,
which is critical under heterogeneous (non-IID) data distributions.

```bash
python train_federated.py --mode server --strategy fedprox --mu 0.1
```

---

## Privacy Mechanisms

### SHA-512 Integrity Verification
Every client attaches a SHA-512 hash of its parameter vector to each
model update. The server verifies and logs these hashes. This ensures
parameters were not accidentally corrupted in transit.

**Limitation**: Does not prevent a malicious client from sending crafted
weights (that requires Secure Aggregation — SecAgg).

### Differential Privacy (optional)
Gradient clipping + Gaussian noise is applied during local training.

```bash
# Enable DP on all clients
python train_federated.py --mode client --client_id 0 --dp --noise 1.0 --clip 1.0
```

**DP guarantees (ε, δ)**:
- ε (epsilon): privacy budget — smaller = more private
- δ (delta): failure probability — typically < 1/n

The server logs the approximate privacy budget consumed each round.

---

## Evaluation Metrics

Accuracy alone is **insufficient** for imbalanced medical data.

| Metric       | Why it matters in healthcare                                    |
|--------------|-----------------------------------------------------------------|
| **AUC-ROC**  | Primary metric — threshold-independent, handles class imbalance |
| **Recall**   | Sensitivity — missed diabetics (false negatives) are dangerous  |
| **Precision**| Positive predictive value — reduces unnecessary alarms          |
| **F1 Score** | Harmonic mean of precision and recall                           |
| **AUC-PR**   | Better than AUC-ROC for highly imbalanced datasets              |
| **Specificity** | Correct identification of non-diabetics                      |

**Threshold selection**: Youden's J statistic (TPR − FPR maximisation) is
used to find the optimal decision threshold rather than a fixed 0.5.

---

## Centralised Baseline

Run the centralized model for comparison:

```bash
python train_federated.py --mode centralized
```

The gap between centralized and federated AUC-ROC measures the cost of federation.

---

## Inference

```bash
# Interactive — prompt for patient data
python inference.py

# With specific model checkpoint
python inference.py --model artifacts/best_model.pth --threshold 0.45

# Batch prediction on a CSV file
python inference.py --batch patients.csv
```

The inference script:
1. Loads the global model and fitted scaler
2. Accepts 8 clinical feature values
3. Applies the same standardisation used during training
4. Returns predicted probability, risk level, and clinical recommendation

---

## Experiment Tracking

All runs are logged to `experiments/runs/` in both TensorBoard and JSON format.

```bash
tensorboard --logdir experiments/runs
```

Logged metrics per round:
- `global/auc_roc`, `global/f1`, `global/recall`, `global/accuracy`
- `global/precision`, `global/specificity`, `global/auc_pr`
- `privacy/epsilon`, `privacy/delta`
- `client_N/auc_roc`, `client_N/recall`, etc.

---

## Results Summary (Expected Range)

| System              | AUC-ROC      | F1          | Recall       |
|---------------------|--------------|-------------|--------------|
| Centralised         | 0.83 – 0.87  | 0.70 – 0.76 | 0.72 – 0.80  |
| FL FedAvg (IID)     | 0.80 – 0.85  | 0.67 – 0.74 | 0.70 – 0.78  |
| FL FedProx (non-IID)| 0.79 – 0.84  | 0.66 – 0.73 | 0.69 – 0.77  |
| FL + DP (σ=1.0)     | 0.76 – 0.82  | 0.63 – 0.70 | 0.66 – 0.74  |

*Actual values depend on random seed, alpha, and number of rounds.*

---

## References

1. McMahan, H. B., et al. (2017). **Communication-efficient learning of deep networks from decentralized data.** AISTATS 2017.
2. Li, T., et al. (2020). **Federated optimization in heterogeneous networks.** MLSys 2020.
3. Abadi, M., et al. (2016). **Deep learning with differential privacy.** CCS 2016.
4. Beutel, D. J., et al. (2020). **Flower: A friendly federated learning research framework.** arXiv:2007.14390.
5. Smith, V., et al. (2017). **Federated multi-task learning.** NeurIPS 2017.
