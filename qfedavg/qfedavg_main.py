#!/usr/bin/env python3
"""
q-FedAvg (PyTorch) — unified experiment runner.

Implements the q-FedAvg algorithm from:
  "Fair Resource Allocation in Federated Learning" (Li et al., 2020)
  https://arxiv.org/abs/1905.10497

Algorithm per round:
  For each selected client i:
    1. Record loss F_i on full local training data (using current global model)
    2. Train locally for E epochs → new weights w_i
    3. Pseudo-gradient:  g_i = (w_before - w_i) / lr
    4. Delta_i = F_i^q  *  g_i          (weighted gradient)
    5. h_i     = q * F_i^(q-1) * ||g_i||^2  +  (1/lr) * F_i^q
  Aggregate:  w_new = w_before - sum(Delta_i) / sum(h_i)

q=0 → reduces to FedAvg; higher q → more fairness pressure.

Usage:
    python qfedavg_main.py --dataset cifar10
    python qfedavg_main.py --dataset cifar100 --model mobilenet --q 5
    python qfedavg_main.py --dataset mnist --model simplemnistcnn --clients_per_round 5 10
    python qfedavg_main.py --dataset cifar10 --num_rounds 5 --clients_per_round 5  # quick test
"""

import os
import sys
import json
import logging
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, mobilenet_v2

from sent140_dataset import load_sent140
from text_models import Sent140LSTM, load_glove_embeddings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


# ── models ────────────────────────────────────────────────────────────────────

class ResNetFed(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc      = nn.Linear(base.fc.in_features, num_classes)
        self.model   = base

    def forward(self, x):
        return self.model(x)


class MobileNetFed(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = mobilenet_v2(weights=None)
        base.features[0][0] = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        base.classifier[1] = nn.Linear(base.last_channel, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


class SimpleMNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, num_classes)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


_NUM_CLASSES   = {"cifar10": 10, "cifar100": 100, "mnist": 10, "sent140": 2}
_DEFAULT_MODEL = {"cifar10": "resnet", "cifar100": "resnet", "mnist": "simplemnistcnn", "sent140": "lstm"}


def build_model(model_name: str, num_classes: int, vocab_size=None, vocab=None):
    if model_name == "resnet":
        return ResNetFed(num_classes=num_classes)
    if model_name == "mobilenet":
        return MobileNetFed(num_classes=num_classes)
    if model_name == "simplemnistcnn":
        return SimpleMNISTCNN(num_classes=num_classes)
    if model_name == "lstm":
        pretrained = None
        if vocab is not None:
            glove_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "data", "glove", "glove.6B.300d.txt"
            )
            pretrained = load_glove_embeddings(glove_path, vocab)
            if pretrained is not None:
                logger.info(f"Loaded GloVe embeddings from {glove_path}")
        return Sent140LSTM(vocab_size=vocab_size, num_classes=num_classes,
                           pretrained_embeddings=pretrained, freeze_embeddings=False)
    raise ValueError(f"Unknown model: {model_name}")


# ── dataset loading ───────────────────────────────────────────────────────────

def load_dataset(name: str):
    if name == "cifar10":
        t_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        t_te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tr = datasets.CIFAR10("../cifar10-fedrl/data", train=True,  download=False, transform=t_tr)
        te = datasets.CIFAR10("../cifar10-fedrl/data", train=False, download=False, transform=t_te)

    elif name == "cifar100":
        t_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        t_te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        tr = datasets.CIFAR100("../data", train=True,  download=False, transform=t_tr)
        te = datasets.CIFAR100("../data", train=False, download=False, transform=t_te)

    elif name == "mnist":
        t_tr = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        t_te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        tr = datasets.MNIST("data/mnist/", train=True,  download=True, transform=t_tr)
        te = datasets.MNIST("data/mnist/", train=False, download=True, transform=t_te)

    elif name == "sent140":
        # Sent140 is loaded per-user in main(); return None here
        return None, None

    else:
        raise ValueError(f"Unknown dataset: {name}")

    logger.info(f"Loaded {name.upper()}: {len(tr)} train, {len(te)} test")
    return tr, te


# ── Dirichlet partition ───────────────────────────────────────────────────────

def dirichlet_partition(dataset, num_clients, num_classes, alpha=0.5,
                         seed=42, min_size=20):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets if hasattr(dataset, "targets")
                      else [y for _, y in dataset])
    idxs = np.arange(len(labels))
    class_idxs = [idxs[labels == c] for c in range(num_classes)]
    for arr in class_idxs:
        rng.shuffle(arr)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        c_idx = class_idxs[c]
        n = len(c_idx)
        if n == 0:
            continue
        p = rng.dirichlet(alpha * np.ones(num_clients))
        counts = np.floor(p * n).astype(int)
        rem = n - counts.sum()
        if rem > 0:
            counts[np.argsort(-(p * n - counts))[:rem]] += 1
        start = 0
        for cid, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[cid].extend(c_idx[start:start + cnt].tolist())
                start += cnt

    if min_size > 0:
        changed = True
        while changed:
            changed = False
            for i in range(num_clients):
                if len(client_indices[i]) < min_size:
                    donor = max(range(num_clients),
                                key=lambda j: len(client_indices[j]))
                    move = min_size - len(client_indices[i])
                    client_indices[i].extend(client_indices[donor][-move:])
                    client_indices[donor] = client_indices[donor][:-move]
                    changed = True

    return [Subset(dataset, idx_list) for idx_list in client_indices]


def bias_partition(dataset, num_clients, num_classes, primary_bias=0.8,
                   seed=42, min_size=20):
    """Bias-based Non-IID: each client gets primary_bias% from one dominant class."""
    rng = np.random.default_rng(seed)
    labels = (np.array(dataset.targets) if hasattr(dataset, "targets")
              else np.array([y for _, y in dataset]))
    class_pools = {c: list(np.where(labels == c)[0]) for c in range(num_classes)}
    for pool in class_pools.values():
        rng.shuffle(pool)
    prefs = (list(range(num_classes)) * (num_clients // num_classes + 1))[:num_clients]
    rng.shuffle(prefs)
    samples_per_client = len(dataset) // num_clients
    majority = int(samples_per_client * primary_bias)
    minority = samples_per_client - majority
    client_indices = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        pref = prefs[i]
        take = min(majority, len(class_pools[pref]))
        client_indices[i].extend(class_pools[pref][:take])
        class_pools[pref] = class_pools[pref][take:]
        others = [c for c in range(num_classes) if c != pref]
        per_other = minority // len(others) if others else 0
        for c in others:
            take = min(per_other, len(class_pools[c]))
            client_indices[i].extend(class_pools[c][:take])
            class_pools[c] = class_pools[c][take:]
    if min_size > 0:
        changed = True
        while changed:
            changed = False
            for i in range(num_clients):
                if len(client_indices[i]) < min_size:
                    donor = max(range(num_clients), key=lambda j: len(client_indices[j]))
                    move = min_size - len(client_indices[i])
                    client_indices[i].extend(client_indices[donor][-move:])
                    client_indices[donor] = client_indices[donor][:-move]
                    changed = True
    return [Subset(dataset, idxs) for idxs in client_indices]


def partition_clients(dataset, num_clients, num_classes, partition, alpha, primary_bias, seed=42):
    if partition == "dirichlet":
        return dirichlet_partition(dataset, num_clients, num_classes,
                                   alpha=alpha, seed=seed, min_size=20)
    return bias_partition(dataset, num_clients, num_classes,
                          primary_bias=primary_bias, seed=seed, min_size=20)


def split_train_test(subsets, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    splits = []
    for s in subsets:
        n = len(s)
        n_test = max(1, int(n * test_ratio))
        perm = rng.permutation(n)
        splits.append((Subset(s, perm[n_test:].tolist()),
                       Subset(s, perm[:n_test].tolist())))
    return splits


# ── fairness helpers ──────────────────────────────────────────────────────────

def jain_fairness_index(accs):
    n = len(accs)
    if n == 0:
        return float("nan")
    s1, s2 = sum(accs), sum(a * a for a in accs)
    return (s1 ** 2) / (n * s2) if s2 > 0 else 1.0


def evaluate_per_client(model, test_subsets, device):
    model.eval()
    accs = []
    with torch.no_grad():
        for subset in test_subsets:
            loader = DataLoader(subset, batch_size=128, shuffle=False)
            correct = total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                _, pred = torch.max(model(x), 1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
            accs.append(correct / total if total > 0 else 0.0)
    return accs


# ── q-FedAvg core ─────────────────────────────────────────────────────────────

def compute_full_loss(model, loader, criterion, device):
    """Compute mean cross-entropy loss over the full dataset (no grad)."""
    model.eval()
    total_loss = total_n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item() * y.size(0)
            total_n    += y.size(0)
    return total_loss / total_n if total_n > 0 else 0.0


def train_local(model, train_subset, criterion, device, local_epochs, batch_size, lr,
                optimizer_name="sgd"):
    """Train model locally; returns updated state_dict."""
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                        drop_last=False)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=1e-4)
    model.train()
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return {k: v.clone() for k, v in model.state_dict().items()}


def qfedavg_round(global_model, selected_train_subsets, criterion, device,
                  local_epochs, batch_size, lr, q, optimizer_name="sgd", eta_s=0.12):
    """One round of q-FedAvg. Updates global_model in-place.

    q-FedAvg aggregation applies only to trainable parameters.
    Non-parameter buffers (BN running stats, num_batches_tracked) are
    averaged separately — including them in grad_norm_sq would make h_i
    astronomically large and freeze the model.

    When q <= 0, falls back to exact FedAvg (average client models directly).
    """
    param_keys   = {k for k, _ in global_model.named_parameters()}
    full_state   = global_model.state_dict()

    if q <= 0:
        # Exact FedAvg: average trained model weights and buffers
        client_models = []
        for train_subset in selected_train_subsets:
            reset_state = dict(full_state)
            global_model.load_state_dict(reset_state)
            weights_after = train_local(global_model, train_subset, criterion,
                                        device, local_epochs, batch_size, lr, optimizer_name)
            client_models.append(weights_after)

        new_state = {}
        for key in full_state:
            new_state[key] = torch.stack(
                [cm[key].float() for cm in client_models], 0
            ).mean(0).to(full_state[key].dtype)
        global_model.load_state_dict(new_state)
        return

    # float copies for arithmetic
    params_before  = {k: v.clone().float() for k, v in full_state.items() if k in param_keys}
    buffers_before = {k: v.clone()         for k, v in full_state.items() if k not in param_keys}

    Deltas           = []
    hs               = []
    client_buffers   = []   # collect per-client buffer states for averaging

    for train_subset in selected_train_subsets:
        # Reset global model to current global state before each client
        reset_state = {**{k: params_before[k].to(full_state[k].dtype) for k in param_keys},
                       **buffers_before}
        global_model.load_state_dict(reset_state)

        # Loss BEFORE local training (on full local data)
        loss_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        F_i = max(float(compute_full_loss(global_model, loss_loader, criterion, device)), 1e-10)

        # Local training
        weights_after = train_local(global_model, train_subset, criterion,
                                    device, local_epochs, batch_size, lr, optimizer_name)

        # Collect buffer states (BN running stats etc.) for later averaging
        client_buffers.append({k: weights_after[k].clone()
                               for k in weights_after if k not in param_keys})

        # Average per-step gradient: g_i = (w_before - w_after) / lr / steps
        num_batches = (len(train_subset) + batch_size - 1) // batch_size
        local_steps = local_epochs * num_batches
        grads = {k: (params_before[k] - weights_after[k].float()) / lr / local_steps
                 for k in param_keys}
        grad_norm_sq = sum((g ** 2).sum().item() for g in grads.values())

        # q-FedAvg formula
        Deltas.append({k: (F_i ** q) * grads[k] for k in param_keys})
        hs.append(q * (F_i ** (q - 1)) * grad_norm_sq + (1.0 / eta_s) * (F_i ** q))

    # Aggregate parameters with q-FedAvg weighting
    denom     = max(sum(hs), 1e-10)
    new_state = {}
    for k in param_keys:
        delta_sum = sum(d[k] for d in Deltas)
        new_val   = params_before[k] - delta_sum / denom
        new_state[k] = new_val.to(full_state[k].dtype)

    # Average buffers across selected clients (standard FedAvg for non-parameters)
    n = len(client_buffers)
    for k in buffers_before:
        try:
            avg = sum(cb[k].float() for cb in client_buffers) / n
            new_state[k] = avg.to(buffers_before[k].dtype)
        except Exception:
            new_state[k] = buffers_before[k]

    global_model.load_state_dict(new_state)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_class_distribution(client_subsets, num_classes, out_path, max_clients=None):
    """Stacked bar chart: each bar = one client, stacked by class distribution."""
    n = len(client_subsets) if max_clients is None else min(len(client_subsets), max_clients)
    counts = np.zeros((n, num_classes), dtype=int)
    for cid in range(n):
        labels = [client_subsets[cid].dataset[i][1] for i in client_subsets[cid].indices]
        c = Counter(labels)
        for cls in range(num_classes):
            counts[cid, cls] = c.get(cls, 0)
    xs = np.arange(n)
    bottoms = np.zeros(n)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.15), 6))
    for cls in range(num_classes):
        ax.bar(xs, counts[:, cls], bottom=bottoms, label=f"Class {cls}")
        bottoms += counts[:, cls]
    ax.set_xlabel("Client")
    ax.set_ylabel("Samples")
    ax.set_title("Per-Client Class Distribution (Stacked)")
    if n > 40:
        step = max(1, n // 20)
        ax.set_xticks(xs[::step])
        ax.set_xticklabels([str(i) for i in xs[::step]])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=min(5, num_classes), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_plots(mean_accuracies, std_accuracies, jfi_scores,
               participation_freq, num_clients, dataset_name, run_plots_path,
               all_per_client_accs=None):
    n = len(mean_accuracies)
    rounds = range(1, n + 1)

    # Mean per-client accuracy with ±1 std shading
    plt.figure(figsize=(10, 6))
    mean_arr = np.array(mean_accuracies)
    std_arr  = np.array(std_accuracies)
    plt.plot(rounds, mean_arr, "b-", linewidth=2, marker="o", label="Mean acc")
    plt.fill_between(rounds, mean_arr - std_arr, mean_arr + std_arr,
                     alpha=0.2, color="blue", label="±1 std")
    plt.title(f"Per-Client Accuracy — {dataset_name.upper()} ({num_clients} clients)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/accuracy_mean_std.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy std deviation over rounds
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, std_arr, "m-", linewidth=2, marker="^")
    plt.title(f"Per-Client Accuracy Std Dev — {dataset_name.upper()}")
    plt.xlabel("Round")
    plt.ylabel("Std Dev")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/accuracy_std.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Jain's Fairness Index over rounds
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, jfi_scores, "g-", linewidth=2, marker="D")
    plt.title(f"Jain's Fairness Index — {dataset_name.upper()}")
    plt.xlabel("Round")
    plt.ylabel("JFI (0=worst, 1=perfect)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/jain_fairness_index.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Participation frequency
    plt.figure(figsize=(10, 6))
    counts = [participation_freq.get(i, 0) for i in range(num_clients)]
    plt.bar(range(num_clients), counts, color="green", alpha=0.7)
    plt.title("Client Participation Frequency")
    plt.xlabel("Client ID")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/participation_freq.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    # 1D scatter of final round per-client accuracies
    if all_per_client_accs is not None and len(all_per_client_accs) > 0:
        final_accs = all_per_client_accs[-1]
        plt.figure(figsize=(12, 2))
        plt.scatter(final_accs, np.zeros_like(final_accs), alpha=0.6, s=30, c="blue")
        plt.xlabel("Client Accuracy")
        plt.yticks([])
        plt.title(f"Final Round Per-Client Accuracy Spread — {dataset_name.upper()}")
        plt.xlim(0, 1)
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{run_plots_path}/per_client_accuracy_spread.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Top 10% vs Bottom 10% client accuracy over rounds
        n10 = max(1, num_clients // 10)
        top10s = []
        bot10s = []
        for round_accs in all_per_client_accs:
            sorted_a = sorted(round_accs)
            top10s.append(sum(sorted_a[-n10:]) / n10)
            bot10s.append(sum(sorted_a[:n10]) / n10)
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, top10s, "g-", linewidth=2, label="Top 10%")
        plt.plot(rounds, bot10s, "r-", linewidth=2, label="Bottom 10%")
        plt.fill_between(rounds, bot10s, top10s, alpha=0.1, color="gray")
        plt.title(f"Top 10% vs Bottom 10% Client Accuracy — {dataset_name.upper()}")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{run_plots_path}/top10_bottom10.png", dpi=300, bbox_inches="tight")
        plt.close()


# ── main training loop ────────────────────────────────────────────────────────

def run_one(k, cfg, train_ds, test_ds, device,
            sent140_train_subsets=None, sent140_test_subsets=None, vocab_size=None, vocab=None):
    dataset_name = cfg["dataset"]
    model_name   = cfg["model"]
    num_clients  = cfg["num_clients"]
    num_rounds   = cfg["num_rounds"]
    num_classes  = _NUM_CLASSES[dataset_name]
    q            = cfg["q"]
    lr           = cfg["lr"]
    local_epochs = cfg["local_epochs"]
    batch_size   = cfg["batch_size"]
    optimizer_name = cfg.get("optimizer", "sgd")
    eta_s        = cfg.get("eta_s", 0.12)

    out_dir = os.path.join(
        cfg["results_dir"],
        f"{num_clients}_clients_{k}_per_round_{dataset_name}"
    )
    os.makedirs(out_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(out_dir, "logs"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info(f"=== k={k} | dataset={dataset_name} | model={model_name} "
                f"| q={q} | rounds={num_rounds} ===")

    # For sent140, use pre-partitioned per-user data. Otherwise, synthetic partition.
    is_sent140 = dataset_name == "sent140"
    if is_sent140:
        train_subsets = sent140_train_subsets
        test_subsets = sent140_test_subsets
    else:
        client_subsets = partition_clients(
            train_ds, num_clients, num_classes,
            partition=cfg.get("partition", "dirichlet"),
            alpha=cfg["dirichlet_alpha"],
            primary_bias=cfg.get("primary_bias", 0.8),
        )

        plot_class_distribution(
            client_subsets, num_classes,
            out_path=os.path.join(out_dir, "client_class_dist.pdf"),
        )

        splits = split_train_test(client_subsets, test_ratio=0.2, seed=42)
        train_subsets = [s[0] for s in splits]
        test_subsets  = [s[1] for s in splits]

    global_model = build_model(model_name, num_classes, vocab_size=vocab_size, vocab=vocab).to(device)
    criterion    = nn.CrossEntropyLoss()

    mean_accuracies = []
    std_accuracies  = []
    jfi_scores      = []
    all_per_client_accs = []
    top10_accuracies = []
    bottom10_accuracies = []
    participation_freq = {}

    rng = np.random.default_rng(42)

    for rnd in range(num_rounds):
        selected = rng.choice(num_clients, size=k, replace=False).tolist()
        for cid in selected:
            participation_freq[cid] = participation_freq.get(cid, 0) + 1

        selected_train = [train_subsets[i] for i in selected]

        qfedavg_round(global_model, selected_train, criterion, device,
                      local_epochs, batch_size, lr, q, optimizer_name, eta_s)

        per_client_acc = evaluate_per_client(global_model, test_subsets, device)
        mean_acc = float(np.mean(per_client_acc))
        std_acc  = float(np.std(per_client_acc))
        jfi      = jain_fairness_index(per_client_acc)

        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)
        jfi_scores.append(jfi)
        all_per_client_accs.append(per_client_acc)
        n10 = max(1, num_clients // 10)
        sorted_accs = sorted(per_client_acc)
        top10_accuracies.append(sum(sorted_accs[-n10:]) / n10)
        bottom10_accuracies.append(sum(sorted_accs[:n10]) / n10)

        logger.info(f"  Round {rnd+1}/{num_rounds} — "
                    f"MeanAcc={mean_acc:.4f}  StdAcc={std_acc:.4f}  JFI={jfi:.4f}")

    logger.info(f"  Final mean acc : {mean_accuracies[-1]:.4f}")
    logger.info(f"  Final JFI      : {jfi_scores[-1]:.4f}")
    logger.info(f"  Best mean acc  : {max(mean_accuracies):.4f}")

    with open(os.path.join(out_dir, "run_results.json"), "w") as f:
        json.dump({
            "config": {
                "dataset":           dataset_name,
                "model":             model_name,
                "num_clients":       num_clients,
                "num_rounds":        num_rounds,
                "clients_per_round": k,
                "q":                 q,
                "lr":                lr,
                "local_epochs":      local_epochs,
                "optimizer":         optimizer_name,
                "eta_s":             eta_s,
                "dirichlet_alpha":   cfg["dirichlet_alpha"],
                "method":            "qfedavg",
            },
            "k":                k,
            "mean_accuracies":  mean_accuracies,
            "std_accuracies":   std_accuracies,
            "jfi_scores":       jfi_scores,
            "participation_freq": {str(c): v
                                   for c, v in participation_freq.items()},
            "per_client_accuracies": all_per_client_accs,
            "top10_accuracies": top10_accuracies,
            "bottom10_accuracies": bottom10_accuracies,
        }, f, indent=2)
    logger.info(f"Saved → {out_dir}/run_results.json")

    plots_path = os.path.join(out_dir, "plots")
    os.makedirs(plots_path, exist_ok=True)
    save_plots(mean_accuracies, std_accuracies, jfi_scores,
               participation_freq, num_clients, dataset_name, plots_path,
               all_per_client_accs=all_per_client_accs)

    logger.removeHandler(fh)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="q-FedAvg experiment runner (PyTorch)")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "mnist", "sent140"],
                   default="cifar10")
    p.add_argument("--model", choices=["resnet", "mobilenet", "simplemnistcnn", "lstm"],
                   default=None,
                   help="Model (default: resnet for CIFAR, simplemnistcnn for MNIST, lstm for Sent140)")
    p.add_argument("--q", type=float, default=7.0,
                   help="Fairness parameter q (0=FedAvg, higher=more fair, default=7)")
    p.add_argument("--num_clients",       type=int,   default=100)
    p.add_argument("--num_rounds",        type=int,   default=200)
    p.add_argument("--clients_per_round", type=int,   nargs="+", default=[5, 10, 20, 30])
    p.add_argument("--local_epochs",      type=int,   default=3)
    p.add_argument("--batch_size",        type=int,   default=128,
                   help="Local training batch size")
    p.add_argument("--lr",                type=float, default=0.01,
                   help="Local optimizer learning rate")
    p.add_argument("--optimizer",         choices=["sgd", "adam"], default="sgd",
                   help="Local optimizer (sgd or adam)")
    p.add_argument("--eta_s",             type=float, default=0.12,
                   help="Server learning rate for q-FedAvg formula (0.12 matches FedAvg step size for lr=0.01, E=3, bs=128)")
    p.add_argument("--dirichlet_alpha",   type=float, default=0.5)
    p.add_argument("--partition",          choices=["dirichlet", "bias"], default="dirichlet",
                   help="Data partition method (default: dirichlet)")
    p.add_argument("--primary_bias",       type=float, default=0.8,
                   help="Dominant class fraction for bias partition (default: 0.8)")
    p.add_argument("--results_dir",       type=str,   default="qfedavg_results")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = args.dataset
    model_name   = args.model or _DEFAULT_MODEL[dataset_name]

    cfg = {
        "dataset":         dataset_name,
        "model":           model_name,
        "q":               args.q,
        "num_clients":     args.num_clients,
        "num_rounds":      args.num_rounds,
        "local_epochs":    args.local_epochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "optimizer":       args.optimizer,
        "eta_s":           args.eta_s,
        "dirichlet_alpha": args.dirichlet_alpha,
        "partition":       args.partition,
        "primary_bias":    args.primary_bias,
        "results_dir":     None,  # set per run below
    }

    logger.info(f"Device={device} | {dataset_name} | {model_name} | q={args.q}")

    train_ds, test_ds = load_dataset(dataset_name)

    sent140_train_subsets = None
    sent140_test_subsets = None
    vocab_size = None
    vocab = None
    if dataset_name == "sent140":
        sent140_json = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "sent140_all_data.json"
        )
        sent140_train_subsets, sent140_test_subsets, vocab = load_sent140(
            sent140_json, num_clients=args.num_clients, min_samples=30, vocab_size=5000, seed=42
        )
        vocab_size = len(vocab)
        actual_clients = len(sent140_train_subsets)
        cfg["num_clients"] = actual_clients
        logger.info(f"Loaded Sent140: {actual_clients} users, vocab size={vocab_size}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root  = os.path.join(args.results_dir, f"{dataset_name}_q{args.q}_{timestamp}")
    os.makedirs(run_root, exist_ok=True)
    cfg["results_dir"] = run_root
    logger.info(f"Output root: {run_root}")

    manifest = {
        "timestamp": timestamp,
        "config": cfg,
        "cli_overrides": {
            k: getattr(args, k) for k in vars(args) if getattr(args, k) is not None
        },
    }
    manifest_path = os.path.join(run_root, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    for k in args.clients_per_round:
        run_one(k, cfg, train_ds, test_ds, device,
                sent140_train_subsets=sent140_train_subsets,
                sent140_test_subsets=sent140_test_subsets,
                vocab_size=vocab_size, vocab=vocab)

    logger.info("All k values complete.")


if __name__ == "__main__":
    main()
