#!/usr/bin/env python3
"""
Per-Client Fairness Experiment — CIFAR-100 with Dirichlet Partition

Motivated by the q-FedAvg paper (Li et al., 2019):
  - Each client holds a local train set AND a local test set
  - After each FL round the global model is evaluated on every client's
    own test set → produces a distribution of per-client accuracies
  - We compare the *variance* (and min/max/mean) of that distribution
    across methods to argue why fairness in client selection matters

Methods compared in a single run:
  1. FedRL   — DQN-based client selection (current agent)
  2. FedAvg  — uniform random selection (baseline)
  3. Ablated — constant epsilon=1.0 (purely random DQN, reward ignored)

Results saved to:
  fairness_results/<method>/run_results.json  (per-round per-client accs)
"""

import os
import sys
import json
import random
import logging
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

from models import ResNetFed
from dqn_agent import DQN_Agent
from environment import FL_Environment
from dirchlet_partitioner import direchlet_partition

# ==================== REPRODUCIBILITY ====================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# ==================== CONFIG ====================
NUM_CLIENTS           = 100
NUM_CLASSES           = 100         # CIFAR-100
NUM_ROUNDS            = 200
CLIENTS_PER_ROUND_LIST = [5, 10, 20, 30]   # iterated over in main()
LOCAL_EPOCHS     = 3
DIRICHLET_ALPHA  = 0.5         # heterogeneity — lower = more non-IID
TEST_SPLIT_RATIO = 0.2         # fraction of each client's data held out as local test
BATCH_SIZE       = 64
RESULTS_BASE     = "fairness_results_multiple_client_sizes"

RL_ALPHA  = 0.5   # KL weight in reward
RL_BETA   = 0.3   # frequency weight in reward

METHODS = ["fedrl", "fedavg", "ablated"]

# ==================== DATA ====================
def load_cifar100():
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
        root="./data", train=True,  download=True, transform=transform_train)
    test_ds  = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test)
    return train_ds, test_ds


def split_client_datasets(client_subsets, test_ratio=0.2, seed=42):
    """
    For each client Subset, split into (train_subset, test_subset).
    Returns list of (train_subset, test_subset) tuples.
    """
    rng = np.random.default_rng(seed)
    splits = []
    for subset in client_subsets:
        indices = list(subset.indices)
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_ratio))
        test_idx  = indices[:n_test]
        train_idx = indices[n_test:]
        train_sub = Subset(subset.dataset, train_idx)
        test_sub  = Subset(subset.dataset, test_idx)
        splits.append((train_sub, test_sub))
    return splits


# ==================== CLIENT ====================
class FairnessClient:
    """Client that holds separate train and test sets."""
    def __init__(self, client_id, train_dataset, test_dataset, num_classes=100):
        self.client_id = client_id
        self.num_classes = num_classes
        self.train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader  = data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetFed(num_classes=num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs=LOCAL_EPOCHS):
        self.model.train()
        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(images), labels)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

    def evaluate_local(self, model_state_dict):
        """Evaluate the given global model on this client's LOCAL test set."""
        tmp = ResNetFed(num_classes=self.num_classes).to(self.device)
        tmp.load_state_dict(model_state_dict)
        tmp.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                _, predicted = torch.max(tmp(images), 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    def get_class_distribution(self):
        counts = np.zeros(self.num_classes)
        for _, label in self.train_loader.dataset:
            counts[label] += 1
        s = counts.sum()
        return counts / s if s > 0 else counts


# ==================== SERVER ====================
class FairnessServer:
    def __init__(self, global_test_loader, num_classes=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = ResNetFed(num_classes=num_classes).to(self.device)
        self.global_test_loader = global_test_loader
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(self, client_state_dicts):
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.stack(
                [sd[key].float() for sd in client_state_dicts], dim=0
            ).mean(0)
        self.global_model.load_state_dict(global_dict)

    def evaluate_global(self):
        self.global_model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.global_test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                total_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total, total_loss / len(self.global_test_loader)


# ==================== SELECTION STRATEGIES ====================
def select_fedavg(num_clients, k):
    return random.sample(range(num_clients), k)


def select_ablated(num_clients, k):
    """Ablated: always random regardless of DQN (epsilon=1)."""
    return random.sample(range(num_clients), k)


def select_fedrl(agent, state, num_clients, k):
    return agent.select_clients(state, num_clients, k)


# ==================== MAIN EXPERIMENT ====================
def run_experiment(method: str, clients, server, env, agent=None, clients_per_round: int = 10):
    """
    Run one FL experiment for `method` in {fedrl, fedavg, ablated}.
    Returns dict with per-round global accuracy and per-round per-client local accuracies.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Starting method: {method.upper()}")
    logger.info(f"{'='*60}")

    # Reset global model weights
    for m in server.global_model.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    # Sync all clients to fresh global model
    gsd = server.global_model.state_dict()
    for c in clients:
        c.model.load_state_dict(copy.deepcopy(gsd))

    global_accuracies = []
    # per_client_accuracies[round_idx] = {client_id: acc}
    per_client_accuracies = []
    participation_freq = {}

    for round_idx in range(NUM_ROUNDS):
        logger.info(f"  [{method}] Round {round_idx+1}/{NUM_ROUNDS}")

        # ---- State (FedRL / ablated use DQN) ----
        if agent is not None:
            state = env.get_state()
        else:
            state = None

        # ---- Client selection ----
        if method == "fedavg":
            selected = select_fedavg(NUM_CLIENTS, clients_per_round)
        else:  # fedrl and ablated both use DQN selection
            selected = select_fedrl(agent, state, NUM_CLIENTS, clients_per_round)

        # ---- Participation tracking ----
        for cid in selected:
            participation_freq[cid] = participation_freq.get(cid, 0) + 1

        # ---- Local training ----
        prev_acc, _ = server.evaluate_global()
        trained_states = []
        for cid in selected:
            sd = clients[cid].train(epochs=LOCAL_EPOCHS)
            trained_states.append(sd)

        # ---- Aggregation ----
        server.aggregate(trained_states)
        gsd = server.global_model.state_dict()

        # Sync all clients
        for c in clients:
            c.model.load_state_dict(copy.deepcopy(gsd))

        # ---- Global evaluation ----
        current_acc, _ = server.evaluate_global()
        global_accuracies.append(current_acc)
        logger.info(f"     Global acc: {current_acc:.4f}")

        # ---- Per-client local evaluation (every round) ----
        round_client_accs = {}
        for c in clients:
            local_acc = c.evaluate_local(gsd)
            round_client_accs[c.client_id] = local_acc
        per_client_accuracies.append(round_client_accs)

        # ---- DQN training (FedRL and ablated) ----
        if agent is not None:
            next_state = env.get_state()
            for cid in selected:
                c = clients[cid]
                part_freq = participation_freq.get(cid, 1)
                if method == "ablated":
                    reward = env.compute_reward_ablated(
                        prev_acc=prev_acc,
                        new_acc=current_acc,
                    )
                else:  # fedrl — full fairness reward
                    reward = env.compute_reward(
                        prev_acc=prev_acc,
                        new_acc=current_acc,
                        client_class_dist=c.get_class_distribution(),
                        client_part_freq=part_freq,
                        client_size=len(c.train_loader.dataset),
                        round=round_idx
                    )
                agent.train(state, cid, reward, next_state)

            state = next_state

    return {
        "global_accuracies": global_accuracies,
        "per_client_accuracies": per_client_accuracies,  # list[dict[int->float]]
        "participation_frequency": participation_freq,
    }


# ==================== ENTRY POINT ====================
def main():
    parser = argparse.ArgumentParser(description="Per-client fairness experiment on CIFAR-100")
    parser.add_argument(
        "--method",
        choices=METHODS + ["all"],
        default="all",
        help="Which method to run: fedrl | fedavg | ablated | all (default: all)",
    )
    args = parser.parse_args()
    methods_to_run = METHODS if args.method == "all" else [args.method]

    os.makedirs(RESULTS_BASE, exist_ok=True)

    # ---- Load CIFAR-100 ----
    logger.info("Loading CIFAR-100...")
    train_ds, test_ds = load_cifar100()

    # ---- Dirichlet partition across clients ----
    logger.info(f"Partitioning {NUM_CLIENTS} clients with Dirichlet α={DIRICHLET_ALPHA}...")
    client_subsets = direchlet_partition(
        dataset=train_ds,
        num_clients=NUM_CLIENTS,
        num_classes=NUM_CLASSES,
        alpha=DIRICHLET_ALPHA,
        seed=42,
        min_size_per_client=20,
    )

    # ---- Split each client into train / local-test ----
    splits = split_client_datasets(client_subsets, test_ratio=TEST_SPLIT_RATIO)

    # ---- Build clients ----
    clients = [
        FairnessClient(i, train_sub, test_sub, num_classes=NUM_CLASSES)
        for i, (train_sub, test_sub) in enumerate(splits)
    ]
    logger.info(f"Created {len(clients)} clients")

    # ---- Global test loader (held-out CIFAR-100 test set) ----
    global_test_loader = data.DataLoader(test_ds, batch_size=128, shuffle=False)

    # ---- Global class distribution (for RL environment) ----
    global_counts = np.zeros(NUM_CLASSES)
    for subset, _ in splits:
        for _, label in subset:
            global_counts[label] += 1
    global_class_dist = global_counts / global_counts.sum()

    # ---- Run each method × each k ----
    for k in CLIENTS_PER_ROUND_LIST:
        logger.info(f"\n{'#'*60}")
        logger.info(f"  clients_per_round = {k}")
        logger.info(f"{'#'*60}")
        for method in methods_to_run:
            out_dir = os.path.join(RESULTS_BASE, method, f"k_{k}")
            os.makedirs(out_dir, exist_ok=True)

            # Fresh server per method × k
            server = FairnessServer(global_test_loader, num_classes=NUM_CLASSES)

            # Agent / env for FedRL and ablated
            agent = None
            env = None
            if method in ("fedrl", "ablated"):
                env = FL_Environment(
                    num_clients=NUM_CLIENTS,
                    global_class_dist=global_class_dist,
                    alpha=RL_ALPHA,
                    beta=RL_BETA,
                )
                agent = DQN_Agent(state_size=NUM_CLASSES, action_size=NUM_CLIENTS)

            results = run_experiment(method, clients, server, env, agent,
                                     clients_per_round=k)

            # Serialise (convert int keys to str for JSON)
            serialisable = {
                "clients_per_round": k,
                "global_accuracies": results["global_accuracies"],
                "participation_frequency": {
                    str(ck): v for ck, v in results["participation_frequency"].items()
                },
                "per_client_accuracies": [
                    {str(cid): acc for cid, acc in rd.items()}
                    for rd in results["per_client_accuracies"]
                ],
            }
            out_path = os.path.join(out_dir, "fairness_results.json")
            with open(out_path, "w") as f:
                json.dump(serialisable, f, indent=2)
            logger.info(f"  Saved {method} k={k} results → {out_path}")

    logger.info("\nAll methods complete.")


if __name__ == "__main__":
    main()
