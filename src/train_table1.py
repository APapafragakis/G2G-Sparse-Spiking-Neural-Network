import os
import sys

# project root στο sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn
from torch.optim import Adam

from models.dense_snn import DenseSNN          # FC SNN :contentReference[oaicite:2]{index=2}
from models.mixer_snn import MixerSNN          # G2GNet (Proposed, Mixer) :contentReference[oaicite:3]{index=3}
from models.er_snn import ERSNN                # ER Random (το νέο που έφτιαξες)
from data.data_fashionmnist import get_fashion_loaders  # :contentReference[oaicite:4]{index=4}
from utils.encoding import rate_encode         # :contentReference[oaicite:5]{index=5}


# ==========================
#  HYPERPARAMETERS
# ==========================

batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024          # ίδιο width με G2GNet
hidden_dim_dense = 447     # περίπου ίδιο #params με sparse G2GNet
num_classes = 10
num_epochs = 20
lr = 1e-3
weight_decay = 1e-4

# G2GNet mixer hyperparams
num_groups = 8
p_intra = 1.0
p_inter = 0.15

# ER random density (ξεκίνα π.χ. 0.2, το ρυθμίζεις αν θες ίδιο #params με Mixer)
p_er_active = 0.20


def select_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("Using CUDA:", torch.cuda.get_device_name(0))
        return dev
    print("Using CPU")
    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0
    correct = 0

    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        spikes = rate_encode(images, T).to(device)  # [T, B, 784]

        optimizer.zero_grad()
        spk_counts = model(spikes)                 # [B, num_classes]

        loss = nn.CrossEntropyLoss()(spk_counts, labels)
        loss.backward()
        optimizer.step()

        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        spikes = rate_encode(images, T).to(device)
        spk_counts = model(spikes)
        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def build_models():
    """
    ΜΟΝΟ τα 4 rows του Table I:

    1) Fully-Connected v1
    2) Fully-Connected v2
    3) ER Random Graph
    4) G2GNet (Proposed) = Mixer
    """
    models_cfg = [
        {
            "id": "fc_v1",
            "label": "Fully-Connected v1 (same #params)",
            "builder": lambda: DenseSNN(input_dim, hidden_dim_dense, num_classes),
        },
        {
            "id": "fc_v2",
            "label": "Fully-Connected v2 (width=1024)",
            "builder": lambda: DenseSNN(input_dim, hidden_dim, num_classes),
        },
        {
            "id": "er",
            "label": "ER Random Graph",
            "builder": lambda: ERSNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                p_active=p_er_active,
            ),
        },
        {
            "id": "g2g_mixer",
            "label": "G2GNet (Proposed, Mixer)",
            "builder": lambda: MixerSNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_groups=num_groups,
                p_intra=p_intra,
                p_inter=p_inter,
            ),
        },
    ]
    return models_cfg


def main():
    device = select_device()
    train_loader, test_loader = get_fashion_loaders(batch_size)

    models_cfg = build_models()
    results = []

    for cfg in models_cfg:
        print("\n" + "=" * 70)
        print(f"Training model: {cfg['label']}  (id={cfg['id']})")
        print("=" * 70)

        model = cfg["builder"]().to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        num_params = count_params(model)
        final_test_acc = None

        for epoch in range(1, num_epochs + 1):
            train_acc = train_one_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            final_test_acc = test_acc

            print(
                f"[{cfg['id']}] Epoch {epoch:02d} | "
                f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
            )

        results.append({
            "id": cfg["id"],
            "label": cfg["label"],
            "test_acc": final_test_acc,
            "params": num_params,
        })

    # Τελικό Table I (SNN έκδοση, μόνο FashionMNIST)
    print("\n" + "#" * 80)
    print("SNN Table I-style results on FashionMNIST")
    print("#" * 80)
    header = f"{'Connectivity Pattern':35s} | {'Test Acc (%)':12s} | {'#Params':>10s}"
    print(header)
    print("-" * len(header))

    for r in results:
        acc_percent = 100.0 * r["test_acc"]
        line = f"{r['label']:35s} | {acc_percent:12.2f} | {r['params']:10d}"
        print(line)


if __name__ == "__main__":
    main()
