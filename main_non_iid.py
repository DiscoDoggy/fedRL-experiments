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

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from client import Client
from server import Server
from environment import FL_Environment
from dqn_agent import DQN_Agent
from models import ResNetFed, SimpleMNISTCNN, SimpleCIFAR10CNN
from non_iid_distributor import NonIIDDataDistributor
from dirchlet_partitioner import direchlet_partition, plot_stacked_client_class_distributions

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
    if args.use_target_network is not None:
        cfg["use_target_network"] = args.use_target_network
    if args.save_checkpoints is not None:
        cfg["save_checkpoints"] = args.save_checkpoints

    return cfg


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Non-IID FedRL experiment runner"
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file.")
    p.add_argument("--dataset", type=str, choices=["cifar10", "mnist"],
                   default=None, help="Dataset to use.")
    p.add_argument("--model",
                   type=str, choices=["resnet", "simplemnistcnn", "simplecifar10cnn"],
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
    p.add_argument("--reward_formula", type=str, choices=["full", "simple"],
                   default=None, help="Reward formula: 'full' or 'simple'.")
    p.add_argument("--use_target_network", type=lambda x: x.lower() == "true",
                   default=None, help="Use Double DQN (true/false).")
    p.add_argument("--save_checkpoints", type=lambda x: x.lower() == "true",
                   default=None, help="Save DQN + server checkpoints (true/false).")
    return p.parse_args()


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int):
    if model_name == "resnet":
        return ResNetFed(num_classes=num_classes)
    elif model_name == "simplemnistcnn":
        return SimpleMNISTCNN(num_classes=num_classes)
    elif model_name == "simplecifar10cnn":
        return SimpleCIFAR10CNN(num_classes=num_classes)
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
            root="./data", train=True, download=True, transform=transform_train
        )
        test_ds = torchvision.datasets.CIFAR10(
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

def save_plots(accuracies, losses, rewards, participation_freq,
               num_clients, dataset_name, run_plots_path):
    n = len(accuracies)
    rounds = range(1, n + 1)

    # Individual: accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, "b-", linewidth=2, marker="o")
    plt.title(f"Global Accuracy — {dataset_name.upper()} ({num_clients} clients)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/global_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Individual: loss
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, losses, "orange", linewidth=2, marker="d")
    plt.title(f"Global Loss — {dataset_name.upper()} ({num_clients} clients)")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/global_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Individual: rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, rewards, "r-", linewidth=2, marker="s")
    plt.title("Rewards")
    plt.xlabel("Round")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_plots_path}/rewards.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Individual: participation frequency
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


# ── main experiment loop ──────────────────────────────────────────────────────

def run_one(k: int, cfg: dict, train_dataset, test_dataset):
    """Run one complete federated experiment for a single k value."""
    num_clients  = cfg["num_clients"]
    num_rounds   = cfg["num_rounds"]
    num_classes  = 10
    alpha        = cfg["alpha"]
    beta         = cfg["beta"]
    dataset_name = cfg["dataset"]

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
    logging.info("Performing bias-based Non-IID data partitioning...")
    distributor = NonIIDDataDistributor(dataset=train_dataset,
                                        num_clients=num_clients)
    client_datasets = distributor.bias_based_distribution()[0]

    # ── global class distribution ─────────────────────────────────────────────
    global_class_counts = np.zeros(num_classes)
    for cd in client_datasets:
        labels = [cd.dataset[idx][1] for idx in cd.indices]
        for lbl in labels:
            global_class_counts[lbl] += 1
    global_class_dist = global_class_counts / np.sum(global_class_counts)

    # ── clients ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clients = []
    for i in range(num_clients):
        client = Client(i, client_datasets[i], num_classes)
        client.model = build_model(cfg["model"], num_classes).to(device)
        client.optimizer = torch.optim.Adam(
            client.model.parameters(), lr=0.001, weight_decay=1e-4
        )
        client.criterion = torch.nn.CrossEntropyLoss()
        clients.append(client)
    logging.info(f"Created {num_clients} clients with model={cfg['model']}")

    # ── server ────────────────────────────────────────────────────────────────
    server = Server(test_dataset, num_classes)
    server.global_model = build_model(cfg["model"], num_classes).to(device)
    logging.info(f"Initialized server with model={cfg['model']}")

    # ── environment + agent ───────────────────────────────────────────────────
    env = FL_Environment(
        num_clients=num_clients,
        global_class_dist=global_class_dist,
        alpha=alpha,
        beta=beta,
        reward_formula=cfg["reward_formula"],
    )
    agent = DQN_Agent(
        state_size=num_classes,
        action_size=num_clients,
        use_target_network=cfg["use_target_network"],
    )
    logging.info(f"Initialized DQN agent (use_target_network="
                 f"{cfg['use_target_network']}) and FL environment")

    # ── training loop ─────────────────────────────────────────────────────────
    rewards, accuracies, losses = [], [], []
    participation_freq  = {}
    client_accuracies   = {}

    logging.info("Starting federated training...")
    for epoch in range(num_rounds):
        logging.info(f"--- Round {epoch + 1}/{num_rounds} ---")

        state = env.get_state()
        selected_idxs = agent.select_clients(state, num_clients, k)
        logging.info(f"Selected clients: {selected_idxs}")

        for idx in selected_idxs:
            participation_freq[idx] = participation_freq.get(idx, 0) + 1

        prev_acc, prev_loss = server.evaluate()

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

        current_acc, current_loss = server.evaluate()
        accuracies.append(current_acc)
        losses.append(current_loss)

        for idx in selected_idxs:
            contrib = (current_acc - prev_acc) / max(prev_acc, 1e-8)
            client_accuracies.setdefault(idx, []).append(contrib)

        logging.info(f"  Global — Acc: {current_acc:.4f}, Loss: {current_loss:.4f}")

        # Reward
        total_reward = 0.0
        for idx in selected_idxs:
            cd = client_datasets[idx]
            labels = [cd.dataset[j][1] for j in cd.indices]
            cc = np.zeros(num_classes)
            for lbl in labels:
                cc[lbl] += 1
            client_class_dist = cc / np.sum(cc)

            total_reward += env.compute_reward(
                prev_acc=prev_acc,
                new_acc=current_acc,
                client_class_dist=client_class_dist,
                client_part_freq=participation_freq.get(idx, 1),
                client_size=len(cd),
            )
        reward = total_reward / len(selected_idxs) if selected_idxs else 0.0
        rewards.append(reward)

        next_state = env.get_state()
        agent.train(state, selected_idxs, reward, next_state)

        logging.info(f"  Reward: {reward:.4f}")

    # ── checkpoints ───────────────────────────────────────────────────────────
    if cfg.get("save_checkpoints", False):
        save_dqn(agent, os.path.join(full_path, "dqn_checkpoint.pt"))
        save_server_model(server, os.path.join(full_path,
                                               "global_model_checkpoint.pt"))

    # ── summary ───────────────────────────────────────────────────────────────
    logging.info("Training complete!")
    logging.info(f"  Final accuracy : {accuracies[-1]:.4f}")
    logging.info(f"  Final loss     : {losses[-1]:.4f}")
    logging.info(f"  Avg reward     : {np.mean(rewards):.4f}")
    logging.info(f"  Best accuracy  : {max(accuracies):.4f}")

    # ── save results JSON ─────────────────────────────────────────────────────
    results = {
        "config": cfg,
        "k": k,
        "accuracies": accuracies,
        "losses": losses,
        "rewards": rewards,
        "participation_freq": participation_freq,
    }
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # ── plots ─────────────────────────────────────────────────────────────────
    save_plots(accuracies, losses, rewards, participation_freq,
               num_clients, dataset_name, plots_path)
    logging.info(f"Plots saved to {plots_path}/")

    logger.removeHandler(fh)


def main():
    args = parse_args()
    cfg = load_config(args)

    logging.info(f"Config: {cfg}")

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
        json.dump(manifest, fh, indent=2)
    logging.info(f"Manifest written to {manifest_path}")

    try:
        for k in cfg["clients_per_round"]:
            run_one(k, cfg, train_dataset, test_dataset)
    except Exception:
        logging.exception("Experiment failed")
        raise


if __name__ == "__main__":
    main()
