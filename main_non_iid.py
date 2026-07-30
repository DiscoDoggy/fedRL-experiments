#!/usr/bin/env python3
"""
Unified Non-IID FedRL entry point.

Usage:
    # Run with a YAML config file:
    python main_non_iid.py --config configs/cifar10.yaml
    python main_non_iid.py --config configs/mnist.yaml

    # Override individual values on the command line:
    python main_non_iid.py --config configs/cifar10.yaml --num_rounds 100
    python main_non_iid.py --config configs/cifar10.yaml \
        --clients_per_round 5 10 --results_dir my_results

    # Run entirely from CLI (no config file):
    python main_non_iid.py --dataset cifar10 --model resnet \
        --results_dir results_cifar --clients_per_round 10
"""

import sys
import os

# Make src/ importable regardless of where the script is invoked from.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import logging
import random
import json
from datetime import datetime

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from client import Client
from server import Server
from environment import FL_Environment
from dqn_agent import DQN_Agent
from models import ResNetFed, MobileNetFed, SimpleMNISTCNN, SimpleCIFAR10CNN, FedFCNet
from non_iid_distributor import NonIIDDataDistributor
from dirchlet_partitioner import direchlet_partition, plot_stacked_client_class_distributions
from femnist_dataset import (
    load_femnist_writers,
    compute_global_class_dist,
    NUM_CLASSES as FEMNIST_NUM_CLASSES,
)

# ── reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


# ── config loading ────────────────────────────────────────────────────────────

DEFAULTS = dict(
    dataset="cifar10",
    model="resnet",
    results_dir="results_for_runs",
    clients_per_round=[5, 10, 20, 30],
    num_clients=100,
    num_rounds=200,
    alpha=0.5,
    beta=0.3,
    reward_formula="full",
    use_target_network=False,
    save_checkpoints=False,
    gamma=2.0,
    dirichlet_alpha=0.5,
    partition="dirichlet",
    primary_bias=0.8,
    no_rl=False,
    # femnist-specific
    femnist_data_dir="./data/femnist",
    femnist_min_samples=50,
    femnist_seed=42,
)


def load_config(args) -> dict:
    """Merge DEFAULTS < yaml_file < CLI flags."""
    cfg = dict(DEFAULTS)

    if args.config:
        if not _YAML_AVAILABLE:
            raise ImportError(
                "pyyaml is required for --config. Install with: pip install pyyaml"
            )
        with open(args.config) as fh:
            file_cfg = yaml.safe_load(fh) or {}
        cfg.update({k: v for k, v in file_cfg.items() if v is not None})

    # CLI overrides (only when explicitly supplied)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.model is not None:
        cfg["model"] = args.model
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir
    if args.clients_per_round is not None:
        cfg["clients_per_round"] = args.clients_per_round
    if args.num_clients is not None:
        cfg["num_clients"] = args.num_clients
    if args.num_rounds is not None:
        cfg["num_rounds"] = args.num_rounds
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    if args.beta is not None:
        cfg["beta"] = args.beta
    if args.reward_formula is not None:
        cfg["reward_formula"] = args.reward_formula
    if args.gamma is not None:
        cfg["gamma"] = args.gamma
    if args.use_target_network is not None:
        cfg["use_target_network"] = args.use_target_network
    if args.save_checkpoints is not None:
        cfg["save_checkpoints"] = args.save_checkpoints
    if args.dirichlet_alpha is not None:
        cfg["dirichlet_alpha"] = args.dirichlet_alpha
    if args.partition is not None:
        cfg["partition"] = args.partition
    if args.primary_bias is not None:
        cfg["primary_bias"] = args.primary_bias
    if args.femnist_data_dir is not None:
        cfg["femnist_data_dir"] = args.femnist_data_dir
    if args.femnist_min_samples is not None:
        cfg["femnist_min_samples"] = args.femnist_min_samples
    if args.femnist_seed is not None:
        cfg["femnist_seed"] = args.femnist_seed
    if args.no_rl is not None:
        cfg["no_rl"] = args.no_rl

    return cfg


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Non-IID FedRL experiment runner"
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file.")
    p.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "mnist", "femnist"],
                   default=None, help="Dataset to use.")
    p.add_argument("--model",
                   type=str, choices=["resnet", "mobilenet", "simplemnistcnn", "simplecifar10cnn", "fedfcnet"],
                   default=None, help="Model architecture.")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Parent directory for result folders.")
    p.add_argument("--clients_per_round", type=int, nargs="+", default=None,
                   help="List of k values, e.g. --clients_per_round 5 10 20 30")
    p.add_argument("--num_clients", type=int, default=None)
    p.add_argument("--num_rounds", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None,
                   help="KL divergence balancing factor.")
    p.add_argument("--beta", type=float, default=None,
                   help="Participation frequency balancing factor.")
    p.add_argument("--reward_formula", type=str,
                   choices=["full", "simple", "fairness", "per_client", "kl_capped", "rank_ema"],
                   default=None,
                   help="Reward formula: 'full', 'simple', 'fairness', 'per_client', 'kl_capped', or 'rank_ema'.")
    p.add_argument("--gamma", type=float, default=None,
                   help="Fairness pressure for 'fairness' reward formula (default: 2.0).")
    p.add_argument("--use_target_network", type=lambda x: x.lower() == "true",
                   default=None, help="Use Double DQN (true/false).")
    p.add_argument("--save_checkpoints", type=lambda x: x.lower() == "true",
                   default=None, help="Save DQN + server checkpoints (true/false).")
    p.add_argument("--dirichlet_alpha", type=float, default=None,
                   help="Dirichlet concentration parameter for non-IID partition (default: 0.5).")
    p.add_argument("--partition", choices=["dirichlet", "bias"], default=None,
                   help="Data partition method (default: dirichlet).")
    p.add_argument("--primary_bias", type=float, default=None,
                   help="Dominant class fraction for bias partition (default: 0.8).")
    p.add_argument("--femnist_data_dir", type=str, default=None,
                   help="Path to F-EMNIST data root (contains train/ and test/).")
    p.add_argument("--femnist_min_samples", type=int, default=None,
                   help="Minimum training samples for a writer to qualify as a client.")
    p.add_argument("--femnist_seed", type=int, default=None,
                   help="Seed controlling which writers are selected as clients.")
    p.add_argument("--no_rl", type=lambda x: x.lower() == "true",
                   default=None, help="Disable RL client selection (uses random selection, FedAvg mode).")
    return p.parse_args()


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int):
    if model_name == "resnet":
        return ResNetFed(num_classes=num_classes)
    elif model_name == "mobilenet":
        return MobileNetFed(num_classes=num_classes)
    elif model_name == "simplemnistcnn":
        return SimpleMNISTCNN(num_classes=num_classes)
    elif model_name == "simplecifar10cnn":
        return SimpleCIFAR10CNN(num_classes=num_classes)
    elif model_name == "fedfcnet":
        return FedFCNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── dataset loading ───────────────────────────────────────────────────────────

def load_dataset(dataset_name: str):
    """Return (train_dataset, test_dataset) for the chosen dataset."""
    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = torchvision.datasets.CIFAR10(
            root="./cifar10-fedrl/data", train=True, download=False, transform=transform_train
        )
        test_ds = torchvision.datasets.CIFAR10(
            root="./cifar10-fedrl/data", train=False, download=False, transform=transform_test
        )

    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        train_ds = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_ds = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset_name == "mnist":
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_ds = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logging.info(
        f"Loaded {dataset_name.upper()} dataset: "
        f"{len(train_ds)} train, {len(test_ds)} test"
    )
    return train_ds, test_ds


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_dqn(agent: DQN_Agent, path: str):
    check_pt = {
        "model": agent.model.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "hparams": {
            "state_size": agent.model.fc1.in_features,
            "action_size": agent.model.fc2.out_features,
            "lr": agent.optimizer.param_groups[0]["lr"],
            "gamma": 0.9,
        },
    }
    torch.save(check_pt, path)


def save_server_model(server: Server, path: str = "global_model.pt"):
    torch.save({"model_state": server.global_model.state_dict()}, path)


# ── plotting ──────────────────────────────────────────────────────────────────

def save_plots(mean_accuracies, std_accuracies, jfi_scores, losses, rewards,
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

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, losses, "orange", linewidth=2, marker="d")
    plt.title(f"Mean Client Test Loss — {dataset_name.upper()} ({num_clients} clients)")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/global_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, rewards, "r-", linewidth=2, marker="s")
    plt.title("Rewards")
    plt.xlabel("Round")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/rewards.png", dpi=300, bbox_inches="tight")
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


# ── main experiment loop ──────────────────────────────────────────────────────

def run_one(k: int, cfg: dict, train_dataset, test_dataset):
    """Run one complete federated experiment for a single k value."""
    num_clients  = cfg["num_clients"]
    num_rounds   = cfg["num_rounds"]
    dataset_name = cfg["dataset"]
    num_classes  = 100 if dataset_name == "cifar100" else 10
    alpha        = cfg["alpha"]
    beta         = cfg["beta"]

    # ── output paths ──────────────────────────────────────────────────────────
    per_run_path = f"{num_clients}_clients_{k}_per_round_{dataset_name}"
    full_path    = os.path.join(cfg["results_dir"], per_run_path)
    plots_path   = os.path.join(full_path, "plots")
    json_path    = os.path.join(full_path, "run_results.json")
    dist_path    = os.path.join(full_path, "client_dist_plots")
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(dist_path, exist_ok=True)

    fh = logging.FileHandler(os.path.join(full_path, "logs"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logging.info(f"=== Starting k={k} | dataset={dataset_name} | "
                 f"model={cfg['model']} | reward={cfg['reward_formula']} ===")
    logging.info(f"  alpha={alpha}, beta={beta}, "
                 f"use_target_network={cfg['use_target_network']}")

    # ── data distribution ─────────────────────────────────────────────────────
    partition = cfg.get("partition", "dirichlet")
    logging.info(f"Performing {partition} Non-IID data partitioning...")
    if partition == "bias":
        distributor = NonIIDDataDistributor(dataset=train_dataset,
                                            num_clients=num_clients,
                                            num_classes=num_classes)
        client_datasets = distributor.bias_based_distribution(
            primary_bias=cfg.get("primary_bias", 0.8)
        )[0]
    else:
        client_datasets = direchlet_partition(
            dataset=train_dataset,
            num_clients=num_clients,
            num_classes=num_classes,
            alpha=cfg.get("dirichlet_alpha", 0.5),
            seed=42,
            min_size_per_client=20,
        )

    # ── per-client class distribution plot ────────────────────────────────────
    if cfg["dataset"] != "femnist":
        plot_stacked_client_class_distributions(
            client_datasets,
            num_classes=num_classes,
            out_path=os.path.join(dist_path, "client_class_dist.pdf"),
        )

    # ── global class distribution ─────────────────────────────────────────────
    global_class_counts = np.zeros(num_classes)
    for cd in client_datasets:
        for idx in cd.indices:
            global_class_counts[int(cd.dataset[idx][1])] += 1
    global_class_dist = global_class_counts / np.sum(global_class_counts)

    # ── clients: split each client's data into 80% train / 20% local test ─────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clients = []
    client_test_datasets = []
    rng = torch.Generator()
    rng.manual_seed(42)
    for i in range(num_clients):
        full_subset = client_datasets[i]
        n_full = len(full_subset)
        n_test = max(1, int(n_full * 0.2))
        n_train = n_full - n_test
        train_subset, test_subset = torch.utils.data.random_split(
            full_subset, [n_train, n_test], generator=rng
        )
        client_test_datasets.append(test_subset)
        client = Client(i, train_subset, num_classes)
        client.model = build_model(cfg["model"], num_classes).to(device)
        client.optimizer = torch.optim.Adam(
            client.model.parameters(), lr=0.001, weight_decay=1e-4
        )
        client.criterion = torch.nn.CrossEntropyLoss()
        clients.append(client)
    logging.info(f"Created {num_clients} clients with model={cfg['model']}")

    # ── server ────────────────────────────────────────────────────────────────
    server = Server(client_test_datasets, num_classes)
    server.global_model = build_model(cfg["model"], num_classes).to(device)
    logging.info(f"Initialized server with model={cfg['model']}")

    # ── environment + agent ───────────────────────────────────────────────────
    env = FL_Environment(
        num_clients=num_clients,
        global_class_dist=global_class_dist,
        alpha=alpha,
        beta=beta,
        reward_formula=cfg["reward_formula"],
        gamma=cfg.get("gamma", 2.0),
    )
    no_rl = cfg.get("no_rl", False)
    if no_rl:
        agent = None
        logging.info("No-RL mode: using random client selection (FedAvg)")
    else:
        agent = DQN_Agent(
            state_size=num_classes,
            action_size=num_clients,
            use_target_network=cfg["use_target_network"],
        )
    logging.info(f"Initialized DQN agent (use_target_network="
                 f"{cfg['use_target_network']}) and FL environment")

    # ── training loop ─────────────────────────────────────────────────────────
    rewards, mean_accuracies, std_accuracies, jfi_scores, losses = [], [], [], [], []
    all_per_client_accs = []
    top10_accuracies = []
    bottom10_accuracies = []
    participation_freq = {}

    logging.info("Starting federated training...")
    for epoch in range(num_rounds):
        logging.info(f"--- Round {epoch + 1}/{num_rounds} ---")

        if no_rl:
            selected_idxs = random.sample(range(num_clients), min(k, num_clients))
        else:
            state = env.get_state()
            selected_idxs = agent.select_clients(state, num_clients, k)
        logging.info(f"Selected clients: {selected_idxs}")

        for idx in selected_idxs:
            participation_freq[idx] = participation_freq.get(idx, 0) + 1

        prev_mean_acc, _, _, prev_per_client_accs, _ = server.evaluate_per_client()

        # Local training
        client_models, client_metrics = [], []
        for cid in selected_idxs:
            logging.info(f"  Training client {cid}...")
            result = clients[cid].train(epochs=3)
            client_models.append(result["model_state"])
            client_metrics.append({
                "client_id": cid,
                "final_loss": result["final_loss"],
                "final_accuracy": result["final_accuracy"],
            })

        for m in client_metrics:
            logging.info(
                f"    Client {m['client_id']}: "
                f"Loss={m['final_loss']:.4f}, Acc={m['final_accuracy']:.4f}"
            )

        server.aggregate_models(client_models)

        # Push aggregated weights to all clients
        global_sd = server.global_model.state_dict()
        for client in clients:
            client.model.load_state_dict(global_sd)

        current_mean_acc, current_std_acc, current_jfi, per_client_accs, current_loss = \
            server.evaluate_per_client()
        mean_accuracies.append(current_mean_acc)
        std_accuracies.append(current_std_acc)
        jfi_scores.append(current_jfi)
        losses.append(current_loss)
        all_per_client_accs.append(per_client_accs)
        n10 = max(1, num_clients // 10)
        sorted_accs = sorted(per_client_accs)
        top10_accuracies.append(sum(sorted_accs[-n10:]) / n10)
        bottom10_accuracies.append(sum(sorted_accs[:n10]) / n10)

        logging.info(
            f"  Global — MeanAcc: {current_mean_acc:.4f}, "
            f"StdAcc: {current_std_acc:.4f}, JFI: {current_jfi:.4f}, "
            f"Loss: {current_loss:.4f}"
        )

        # Update rank EMA (no-op for non-rank_ema formulas)
        env.update_history(per_client_accs)

        # Reward
        if no_rl:
            rewards.append(0.0)
        else:
            new_formulas = ('per_client', 'kl_capped', 'rank_ema')
            per_client_rewards = []
            for idx in selected_idxs:
                cd = client_datasets[idx]
                cc = np.zeros(num_classes)
                for i in cd.indices:
                    cc[int(cd.dataset[i][1])] += 1
                client_class_dist = cc / np.sum(cc)

                if cfg["reward_formula"] in new_formulas:
                    local_delta = per_client_accs[idx] - prev_per_client_accs[idx]
                else:
                    local_delta = None

                r = env.compute_reward(
                    prev_acc=prev_mean_acc,
                    new_acc=current_mean_acc,
                    client_class_dist=client_class_dist,
                    client_part_freq=participation_freq.get(idx, 1),
                    client_size=len(cd),
                    client_acc=per_client_accs[idx] if cfg["reward_formula"] in ("fairness", *new_formulas) else None,
                    mean_acc=current_mean_acc if cfg["reward_formula"] in ("fairness", *new_formulas) else None,
                    client_local_delta=local_delta,
                    client_id=idx,
                )
                per_client_rewards.append(r)
            reward = sum(per_client_rewards) / len(per_client_rewards) if per_client_rewards else 0.0
            rewards.append(reward)

            next_state = env.get_state()
            agent.train(state, selected_idxs, per_client_rewards, next_state)

            logging.info(f"  Reward: {reward:.4f}")

    # ── checkpoints ───────────────────────────────────────────────────────────
    if cfg.get("save_checkpoints", False) and not no_rl:
        save_dqn(agent, os.path.join(full_path, "dqn_checkpoint.pt"))
        save_server_model(server, os.path.join(full_path,
                                               "global_model_checkpoint.pt"))

    # ── summary ───────────────────────────────────────────────────────────────
    logging.info("Training complete!")
    logging.info(f"  Final mean acc : {mean_accuracies[-1]:.4f}")
    logging.info(f"  Final std acc  : {std_accuracies[-1]:.4f}")
    logging.info(f"  Final JFI      : {jfi_scores[-1]:.4f}")
    logging.info(f"  Final loss     : {losses[-1]:.4f}")
    logging.info(f"  Avg reward     : {np.mean(rewards):.4f}")
    logging.info(f"  Best mean acc  : {max(mean_accuracies):.4f}")

    # ── save results JSON ─────────────────────────────────────────────────────
    results = {
        "config": cfg,
        "k": k,
        "mean_accuracies": mean_accuracies,
        "std_accuracies": std_accuracies,
        "jfi_scores": jfi_scores,
        "losses": losses,
        "rewards": rewards,
        "participation_freq": participation_freq,
        "per_client_accuracies": all_per_client_accs,
        "top10_accuracies": top10_accuracies,
        "bottom10_accuracies": bottom10_accuracies,
    }
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)

    # ── plots ─────────────────────────────────────────────────────────────────
    save_plots(mean_accuracies, std_accuracies, jfi_scores, losses, rewards,
               participation_freq, num_clients, dataset_name, plots_path,
               all_per_client_accs=all_per_client_accs)
    logging.info(f"Plots saved to {plots_path}/")

    logger.removeHandler(fh)


# ── femnist experiment (writer-partitioned) ───────────────────────────────────

def run_one_femnist(k: int, cfg: dict):
    """Run one FedRL experiment on F-EMNIST with per-writer clients."""
    num_clients  = cfg["num_clients"]
    num_rounds   = cfg["num_rounds"]
    num_classes  = FEMNIST_NUM_CLASSES   # 62
    alpha        = cfg["alpha"]
    beta         = cfg["beta"]

    # ── output paths ──────────────────────────────────────────────────────────
    per_run_path = f"{num_clients}_clients_{k}_per_round_femnist"
    full_path    = os.path.join(cfg["results_dir"], per_run_path)
    plots_path   = os.path.join(full_path, "plots")
    json_path    = os.path.join(full_path, "run_results.json")
    os.makedirs(plots_path, exist_ok=True)

    fh = logging.FileHandler(os.path.join(full_path, "logs"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logging.info(f"=== Starting k={k} | dataset=femnist | "
                 f"model={cfg['model']} | reward={cfg['reward_formula']} ===")
    logging.info(f"  alpha={alpha}, beta={beta}, "
                 f"  femnist_min_samples={cfg['femnist_min_samples']}, "
                 f"  femnist_seed={cfg['femnist_seed']}")

    # ── load per-writer datasets ───────────────────────────────────────────────
    logging.info("Loading F-EMNIST per-writer datasets...")
    writer_ids, train_datasets, test_datasets = load_femnist_writers(
        data_dir=cfg["femnist_data_dir"],
        min_samples=cfg["femnist_min_samples"],
        num_clients=num_clients,
        seed=cfg["femnist_seed"],
    )
    logging.info(f"Selected {len(writer_ids)} writers as clients")

    # ── global class distribution ─────────────────────────────────────────────
    global_class_dist = compute_global_class_dist(train_datasets, num_classes)

    # ── clients ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clients = []
    for i, train_ds in enumerate(train_datasets):
        client = Client(i, train_ds, num_classes)
        client.model = build_model(cfg["model"], num_classes).to(device)
        client.optimizer = torch.optim.Adam(
            client.model.parameters(), lr=0.001, weight_decay=1e-4
        )
        client.criterion = torch.nn.CrossEntropyLoss()
        clients.append(client)
    logging.info(f"Created {num_clients} clients with model={cfg['model']}")

    # ── server — uses per-writer test sets ───────────────────────────────────
    server = Server(test_datasets, num_classes)
    server.global_model = build_model(cfg["model"], num_classes).to(device)
    logging.info(f"Initialized server with model={cfg['model']}")

    # ── environment + agent ───────────────────────────────────────────────────
    env = FL_Environment(
        num_clients=num_clients,
        global_class_dist=global_class_dist,
        alpha=alpha,
        beta=beta,
        reward_formula=cfg["reward_formula"],
        gamma=cfg.get("gamma", 2.0),
    )
    no_rl = cfg.get("no_rl", False)
    if no_rl:
        agent = None
        logging.info("No-RL mode: using random client selection (FedAvg)")
    else:
        agent = DQN_Agent(
            state_size=num_classes,
            action_size=num_clients,
            use_target_network=cfg["use_target_network"],
        )

    # ── pre-compute per-client class distributions (fixed per writer) ─────────
    client_class_dists = []
    for ds in train_datasets:
        cc = np.zeros(num_classes)
        for lbl in ds.y.numpy():
            cc[int(lbl)] += 1
        client_class_dists.append(cc / cc.sum())

    # ── training loop ─────────────────────────────────────────────────────────
    rewards, mean_accuracies, std_accuracies, jfi_scores, losses = [], [], [], [], []
    all_per_client_accs = []
    top10_accuracies = []
    bottom10_accuracies = []
    participation_freq = {}

    logging.info("Starting federated training...")
    for epoch in range(num_rounds):
        logging.info(f"--- Round {epoch + 1}/{num_rounds} ---")

        if no_rl:
            selected_idxs = random.sample(range(num_clients), min(k, num_clients))
        else:
            state = env.get_state()
            selected_idxs = agent.select_clients(state, num_clients, k)
        logging.info(f"Selected clients: {selected_idxs}")

        for idx in selected_idxs:
            participation_freq[idx] = participation_freq.get(idx, 0) + 1

        prev_mean_acc, _, _, prev_per_client_accs, _ = server.evaluate_per_client()

        client_models, client_metrics = [], []
        for cid in selected_idxs:
            logging.info(f"  Training client {cid} (writer {writer_ids[cid]})...")
            result = clients[cid].train(epochs=3)
            client_models.append(result["model_state"])
            client_metrics.append({
                "client_id": cid,
                "writer_id": writer_ids[cid],
                "final_loss": result["final_loss"],
                "final_accuracy": result["final_accuracy"],
            })

        for m in client_metrics:
            logging.info(
                f"    Client {m['client_id']} ({m['writer_id']}): "
                f"Loss={m['final_loss']:.4f}, Acc={m['final_accuracy']:.4f}"
            )

        server.aggregate_models(client_models)
        global_sd = server.global_model.state_dict()
        for client in clients:
            client.model.load_state_dict(global_sd)

        current_mean_acc, current_std_acc, current_jfi, per_client_accs, current_loss = \
            server.evaluate_per_client()
        mean_accuracies.append(current_mean_acc)
        std_accuracies.append(current_std_acc)
        jfi_scores.append(current_jfi)
        losses.append(current_loss)
        all_per_client_accs.append(per_client_accs)
        n10 = max(1, num_clients // 10)
        sorted_accs = sorted(per_client_accs)
        top10_accuracies.append(sum(sorted_accs[-n10:]) / n10)
        bottom10_accuracies.append(sum(sorted_accs[:n10]) / n10)

        logging.info(
            f"  Global — MeanAcc: {current_mean_acc:.4f}, "
            f"StdAcc: {current_std_acc:.4f}, JFI: {current_jfi:.4f}, "
            f"Loss: {current_loss:.4f}"
        )

        env.update_history(per_client_accs)

        if no_rl:
            rewards.append(0.0)
        else:
            new_formulas = ('per_client', 'kl_capped', 'rank_ema')
            per_client_rewards = []
            for idx in selected_idxs:
                if cfg["reward_formula"] in new_formulas:
                    local_delta = per_client_accs[idx] - prev_per_client_accs[idx]
                else:
                    local_delta = None

                r = env.compute_reward(
                    prev_acc=prev_mean_acc,
                    new_acc=current_mean_acc,
                    client_class_dist=client_class_dists[idx],
                    client_part_freq=participation_freq.get(idx, 1),
                    client_size=len(train_datasets[idx]),
                    client_acc=per_client_accs[idx] if cfg["reward_formula"] in ("fairness", *new_formulas) else None,
                    mean_acc=current_mean_acc if cfg["reward_formula"] in ("fairness", *new_formulas) else None,
                    client_local_delta=local_delta,
                    client_id=idx,
                )
                per_client_rewards.append(r)
            reward = sum(per_client_rewards) / len(per_client_rewards) if per_client_rewards else 0.0
            rewards.append(reward)

            next_state = env.get_state()
            agent.train(state, selected_idxs, per_client_rewards, next_state)
            logging.info(f"  Reward: {reward:.4f}")

    # ── checkpoints ───────────────────────────────────────────────────────────
    if cfg.get("save_checkpoints", False) and not no_rl:
        save_dqn(agent, os.path.join(full_path, "dqn_checkpoint.pt"))
        save_server_model(server, os.path.join(full_path,
                                               "global_model_checkpoint.pt"))

    # ── summary + save ────────────────────────────────────────────────────────
    logging.info("Training complete!")
    logging.info(f"  Final mean acc : {mean_accuracies[-1]:.4f}")
    logging.info(f"  Final std acc  : {std_accuracies[-1]:.4f}")
    logging.info(f"  Final JFI      : {jfi_scores[-1]:.4f}")
    logging.info(f"  Final loss     : {losses[-1]:.4f}")
    logging.info(f"  Avg reward     : {np.mean(rewards):.4f}")
    logging.info(f"  Best mean acc  : {max(mean_accuracies):.4f}")

    results = {
        "config": cfg,
        "k": k,
        "writer_ids": writer_ids,
        "mean_accuracies": mean_accuracies,
        "std_accuracies": std_accuracies,
        "jfi_scores": jfi_scores,
        "losses": losses,
        "rewards": rewards,
        "participation_freq": participation_freq,
        "per_client_accuracies": all_per_client_accs,
        "top10_accuracies": top10_accuracies,
        "bottom10_accuracies": bottom10_accuracies,
    }
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)

    save_plots(mean_accuracies, std_accuracies, jfi_scores, losses, rewards,
               participation_freq, num_clients, "femnist", plots_path,
               all_per_client_accs=all_per_client_accs)
    logging.info(f"Plots saved to {plots_path}/")

    logger.removeHandler(fh)


def main():
    args = parse_args()
    cfg = load_config(args)

    logging.info(f"Config: {cfg}")

    # For femnist, data is loaded per-writer inside run_one_femnist.
    train_dataset, test_dataset = None, None
    if cfg["dataset"] != "femnist":
        train_dataset, test_dataset = load_dataset(cfg["dataset"])

    # Stamp the results_dir with a timestamp so re-runs never overwrite.
    # e.g. results_for_runs_cifar_testing/run_20260610_143022/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["results_dir"] = os.path.join(cfg["results_dir"], f"run_{timestamp}")
    os.makedirs(cfg["results_dir"], exist_ok=True)
    logging.info(f"Output directory: {cfg['results_dir']}")

    # Write manifest at the run root so the config is always co-located
    # with the results, regardless of which config file was used.
    manifest = {
        "timestamp": timestamp,
        "config_file": args.config,
        "config": cfg,
        "cli_overrides": {
            k: getattr(args, k)
            for k in vars(args)
            if k != "config" and getattr(args, k) is not None
        },
    }
    manifest_path = os.path.join(cfg["results_dir"], "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, cls=NumpyEncoder)
    logging.info(f"Manifest written to {manifest_path}")

    try:
        for k in cfg["clients_per_round"]:
            if cfg["dataset"] == "femnist":
                run_one_femnist(k, cfg)
            else:
                run_one(k, cfg, train_dataset, test_dataset)
    except Exception:
        logging.exception("Experiment failed")
        raise


if __name__ == "__main__":
    main()
