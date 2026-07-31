#!/usr/bin/env python3
"""
Unified FAVOR experiment runner.

FAVOR: Federated Averaging with Variational Optimal Rewards
Implementation based on Server_FAVOR.py from FLASH-RL codebase.

Paper: "FAVOR: Federated Learning with Variational Optimal Rewards"
(Reference implementation from the FLASH-RL repository)

Usage:
    python favor_main.py --dataset cifar10
    python favor_main.py --dataset cifar100
    python favor_main.py --dataset mnist --model simplemnistcnn
"""

import os
import json
import logging
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, mobilenet_v2

import serverFL.Server_FAVOR as Server_FAVOR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


# ── models ────────────────────────────────────────────────────────────────────

class ResNetFed(nn.Module):
    """Modified ResNet-18 for 32×32 images."""
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc     = nn.Linear(base.fc.in_features, num_classes)
        self.model  = base

    def forward(self, x):
        return self.model(x)


class MobileNetFed(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = mobilenet_v2(weights=None)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

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


_DATASET_NUM_CLASSES   = {"cifar10": 10, "cifar100": 100, "mnist": 10}
_DATASET_DEFAULT_MODEL = {"cifar10": "resnet", "cifar100": "resnet", "mnist": "simplemnistcnn"}


def build_model(model_name: str, num_classes: int):
    if model_name == "resnet":
        return ResNetFed(num_classes=num_classes)
    if model_name == "mobilenet":
        return MobileNetFed(num_classes=num_classes)
    if model_name == "simplemnistcnn":
        return SimpleMNISTCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")


# ── dataset loading ───────────────────────────────────────────────────────────

def load_dataset(dataset_name: str):
    if dataset_name == "cifar10":
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        t_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10("../cifar10-fedrl/data", train=True,  download=False, transform=t_train)
        test_ds  = datasets.CIFAR10("../cifar10-fedrl/data", train=False, download=False, transform=t_test)

    elif dataset_name == "cifar100":
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        t_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_ds = datasets.CIFAR100("../data", train=True,  download=False, transform=t_train)
        test_ds  = datasets.CIFAR100("../data", train=False, download=False, transform=t_test)

    elif dataset_name == "mnist":
        t_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        t_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST("data/mnist/", train=True,  download=True, transform=t_train)
        test_ds  = datasets.MNIST("data/mnist/", train=False, download=True, transform=t_test)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Loaded {dataset_name.upper()}: {len(train_ds)} train, {len(test_ds)} test")
    return train_ds, test_ds


# ── Dirichlet partition ───────────────────────────────────────────────────────

def dirichlet_partition(dataset, num_clients, num_classes, alpha=0.5,
                         seed=42, min_size_per_client=20):
    def _labels(ds):
        if hasattr(ds, "targets"):
            return np.array(ds.targets)
        return np.array([y for _, y in ds])

    rng    = np.random.default_rng(seed)
    labels = _labels(dataset)
    idxs   = np.arange(len(labels))
    class_idxs = [idxs[labels == c] for c in range(num_classes)]
    for arr in class_idxs:
        rng.shuffle(arr)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        c_idx = class_idxs[c]
        n = len(c_idx)
        if n == 0:
            continue
        p      = rng.dirichlet(alpha * np.ones(num_clients))
        counts = np.floor(p * n).astype(int)
        rem    = n - counts.sum()
        if rem > 0:
            counts[np.argsort(-(p * n - counts))[:rem]] += 1
        start = 0
        for cid, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[cid].extend(c_idx[start:start + cnt].tolist())
                start += cnt

    if min_size_per_client > 0:
        changed = True
        while changed:
            changed = False
            for i in range(num_clients):
                if len(client_indices[i]) < min_size_per_client:
                    donor = max(range(num_clients), key=lambda j: len(client_indices[j]))
                    move  = min_size_per_client - len(client_indices[i])
                    client_indices[i].extend(client_indices[donor][-move:])
                    client_indices[donor] = client_indices[donor][:-move]
                    changed = True

    return [Subset(dataset, idx_list) for idx_list in client_indices]


def bias_partition(dataset, num_clients, num_classes, primary_bias=0.8,
                   seed=42, min_size_per_client=20):
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

    if min_size_per_client > 0:
        changed = True
        while changed:
            changed = False
            for i in range(num_clients):
                if len(client_indices[i]) < min_size_per_client:
                    donor = max(range(num_clients), key=lambda j: len(client_indices[j]))
                    move = min_size_per_client - len(client_indices[i])
                    client_indices[i].extend(client_indices[donor][-move:])
                    client_indices[donor] = client_indices[donor][:-move]
                    changed = True

    return [Subset(dataset, idxs) for idxs in client_indices]


def partition_clients(dataset, num_clients, num_classes, partition, alpha, primary_bias, seed=42):
    if partition == "dirichlet":
        return dirichlet_partition(dataset, num_clients, num_classes,
                                   alpha=alpha, seed=seed, min_size_per_client=20)
    else:
        return bias_partition(dataset, num_clients, num_classes,
                              primary_bias=primary_bias, seed=seed, min_size_per_client=20)


def split_train_test(subsets, test_ratio=0.2, seed=42):
    rng    = np.random.default_rng(seed)
    splits = []
    for subset in subsets:
        n      = len(subset)
        n_test = max(1, int(n * test_ratio))
        perm   = rng.permutation(n)
        splits.append((
            Subset(subset, perm[n_test:].tolist()),
            Subset(subset, perm[:n_test].tolist()),
        ))
    return splits


# ── fairness helpers ──────────────────────────────────────────────────────────

def jain_fairness_index(accs):
    n = len(accs)
    if n == 0:
        return float("nan")
    s1 = sum(accs)
    s2 = sum(a * a for a in accs)
    return (s1 ** 2) / (n * s2) if s2 > 0 else 1.0


def evaluate_per_client(model, test_subsets, device):
    model.eval()
    accs = []
    with torch.no_grad():
        for subset in test_subsets:
            loader = DataLoader(subset, batch_size=128, shuffle=False)
            correct = total = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                correct += (predicted == labels).sum().item()
                total   += labels.size(0)
            accs.append(correct / total if total > 0 else 0.0)
    return accs


# ── plotting ──────────────────────────────────────────────────────────────────

def save_plots(mean_accuracies, std_accuracies, jfi_scores,
               participation_freq, num_clients, dataset_name, run_plots_path,
               all_per_client_accs=None):
    n = len(mean_accuracies)
    rounds = range(1, n + 1)

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

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, std_arr, "m-", linewidth=2, marker="^")
    plt.title(f"Per-Client Accuracy Std Dev — {dataset_name.upper()}")
    plt.xlabel("Round")
    plt.ylabel("Std Dev")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/accuracy_std.png", dpi=300, bbox_inches="tight")
    plt.close()

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

    counts = [participation_freq.get(i, 0) for i in range(num_clients)]
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_clients), counts, color="green", alpha=0.7)
    plt.title("Client Participation Frequency")
    plt.xlabel("Client ID")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/participation_freq.png", dpi=300, bbox_inches="tight")
    plt.close()

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Unified FAVOR experiment runner")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "mnist"],
                   default="cifar10")
    p.add_argument("--model", choices=["resnet", "mobilenet", "simplemnistcnn"], default=None)
    p.add_argument("--num_clients",       type=int, default=100)
    p.add_argument("--num_rounds",        type=int, default=200)
    p.add_argument("--clients_per_round", type=int, nargs="+", default=[5, 10, 20, 30])
    p.add_argument("--dirichlet_alpha",   type=float, default=0.5)
    p.add_argument("--partition",          choices=["dirichlet", "bias"], default="dirichlet")
    p.add_argument("--primary_bias",       type=float, default=0.8)
    p.add_argument("--M",                  type=float, default=2.0,
                   help="FAVOR parameter M (base for reward: M^(acc - omega) - 1)")
    p.add_argument("--omega",             type=float, default=0.5,
                   help="FAVOR parameter omega (threshold for reward)")
    p.add_argument("--results_dir",       type=str, default="favor_results_unified")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset
    num_classes  = _DATASET_NUM_CLASSES[dataset_name]
    model_name   = args.model or _DATASET_DEFAULT_MODEL[dataset_name]
    num_clients  = args.num_clients
    num_rounds   = args.num_rounds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device={device} | dataset={dataset_name} | model={model_name} "
                f"| num_classes={num_classes} | num_clients={num_clients} "
                f"| num_rounds={num_rounds} | M={args.M} | omega={args.omega}")

    train_ds, test_ds = load_dataset(dataset_name)

    logger.info(f"Partitioning {num_clients} clients — method={args.partition}...")
    client_subsets = partition_clients(
        train_ds, num_clients, num_classes,
        partition=args.partition,
        alpha=args.dirichlet_alpha,
        primary_bias=args.primary_bias,
    )
    splits = split_train_test(client_subsets, test_ratio=0.2, seed=42)
    test_subsets = [s[1] for s in splits]
    logger.info(f"Partition done — client train sizes: "
                f"min={min(len(s[0]) for s in splits)}, "
                f"max={max(len(s[0]) for s in splits)}")

    rng_hw = np.random.default_rng(0)
    clients_info = [
        (f"client_{i}",
         len(splits[i][0]),
         int(rng_hw.integers(2, 8)),
         [float(rng_hw.uniform(1.0, 3.0))],
         [float(rng_hw.uniform(10.0, 100.0))])
        for i in range(num_clients)
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(args.results_dir, f"{dataset_name}_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Output root: {run_dir}")

    manifest = {
        "timestamp": timestamp,
        "config": {
            "dataset":            dataset_name,
            "model":              model_name,
            "num_clients":        num_clients,
            "num_rounds":         num_rounds,
            "clients_per_round":  args.clients_per_round,
            "dirichlet_alpha":    args.dirichlet_alpha,
            "partition":          args.partition,
            "primary_bias":       args.primary_bias,
            "M":                  args.M,
            "omega":              args.omega,
            "method":             "favor",
        },
        "cli_overrides": {
            k: getattr(args, k) for k in vars(args) if getattr(args, k) is not None
        },
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    for k in args.clients_per_round:
        logger.info(f"\n{'#'*60}\n  k = {k}\n{'#'*60}")

        out_dir = os.path.join(run_dir, f"{num_clients}_clients_{k}_per_round_{dataset_name}")
        os.makedirs(out_dir, exist_ok=True)

        dict_clients = {f"client_{i}": splits[i][0] for i in range(num_clients)}
        global_model = build_model(model_name, num_classes).to(device)

        server = Server_FAVOR.Server_FedDRL(
            num_clients  = num_clients,
            global_model = global_model,
            dict_clients = dict_clients,
            loss_fct     = nn.CrossEntropyLoss(),
            B            = 128,
            dataset_test = test_ds,
            learning_rate= 0.001,
            momentum     = 0.9,
            clients_info = clients_info,
            device       = device,
            per_client_test_subsets = test_subsets,
        )

        results = server.global_train(
            comms_round   = num_rounds,
            C             = k / num_clients,
            E             = 3,
            mu            = 0,
            M             = args.M,
            omega         = args.omega,
            batch_size    = 32,
            verbose_test  = 1,
            verbos        = 1,
        )

        global_accuracies = [
            acc.cpu().item() if isinstance(acc, torch.Tensor) else float(acc)
            for acc in results["Accuracy"]
        ]

        participation_freq = {}
        for round_clients in results.get("Selected_clients", []):
            for cid in round_clients:
                participation_freq[int(cid)] = participation_freq.get(int(cid), 0) + 1

        best_model = build_model(model_name, num_classes).to(device)
        best_model.load_state_dict(results["Best_model_weights"])
        per_client_final = evaluate_per_client(best_model, test_subsets, device)

        mean_acc = float(np.mean(per_client_final))
        std_acc  = float(np.std(per_client_final))
        jfi      = jain_fairness_index(per_client_final)

        per_round_pc = results.get("per_client_accuracies", [])
        if per_round_pc and len(per_round_pc) > 0:
            per_round_pc_clean = [
                [float(a) for a in round_accs] for round_accs in per_round_pc
            ]
            n10 = max(1, num_clients // 10)
            top10_accs = []
            bot10_accs = []
            for round_accs in per_round_pc_clean:
                sorted_a = sorted(round_accs)
                top10_accs.append(sum(sorted_a[-n10:]) / n10)
                bot10_accs.append(sum(sorted_a[:n10]) / n10)
        else:
            per_round_pc_clean = []
            top10_accs = []
            bot10_accs = []

        logger.info(f"  Global acc (final round)  : {global_accuracies[-1]:.4f}")
        logger.info(f"  Per-client mean acc (best): {mean_acc:.4f}")
        logger.info(f"  Per-client std acc  (best): {std_acc:.4f}")
        logger.info(f"  Jain's Fairness Index     : {jfi:.4f}")

        with open(os.path.join(out_dir, "run_results.json"), "w") as f:
            json.dump({
                "config": {
                    "dataset":            dataset_name,
                    "model":              model_name,
                    "num_clients":        num_clients,
                    "num_rounds":         num_rounds,
                    "clients_per_round":  k,
                    "dirichlet_alpha":    args.dirichlet_alpha,
                    "M":                  args.M,
                    "omega":              args.omega,
                    "method":             "favor",
                },
                "k":                          k,
                "global_accuracies":          global_accuracies,
                "final_mean_accuracy":        mean_acc,
                "final_std_accuracy":         std_acc,
                "final_jfi":                  jfi,
                "final_per_client_accuracies": {
                    str(i): acc for i, acc in enumerate(per_client_final)
                },
                "participation_freq": {
                    str(c): v for c, v in participation_freq.items()
                },
                "per_client_accuracies":      per_round_pc_clean,
                "top10_accuracies":           top10_accs,
                "bottom10_accuracies":        bot10_accs,
            }, f, indent=2)
        logger.info(f"Saved → {out_dir}/run_results.json")

        if per_round_pc_clean:
            per_round_means = [sum(pc) / len(pc) for pc in per_round_pc_clean]
            per_round_stds = [float(np.std(pc)) for pc in per_round_pc_clean]
            per_round_jfis = [jain_fairness_index(pc) for pc in per_round_pc_clean]
        else:
            per_round_means = []
            per_round_stds = []
            per_round_jfis = []

        plots_path = os.path.join(out_dir, "plots")
        os.makedirs(plots_path, exist_ok=True)
        save_plots(per_round_means, per_round_stds, per_round_jfis,
                   participation_freq, num_clients, dataset_name, plots_path,
                   all_per_client_accs=per_round_pc_clean)

    logger.info("\nAll k values complete.")


if __name__ == "__main__":
    main()
