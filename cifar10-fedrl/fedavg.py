"""
Flower + PyTorch federated learning on CIFAR‑10
- 100 total clients in the pool
- 60 clients sampled per round (FedAvg: fraction_fit=0.6)

Usage:
  pip install flwr==1.* torch torchvision
  python fl_cifar10_fedavg.py --rounds 5 --local-epochs 1 --batch-size 64 --partition iid

Notes:
- This uses Flower's simulation API (single process by default). For large runs, consider
  installing Ray (`pip install ray`) and setting --use-ray to leverage parallelism.
- Set --device "cuda" to use a GPU if available (or leave to auto-detect).
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

import flwr as fl

# ------------------------------
# Utilities
# ------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

transform_train_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train_cifar
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test_cifar
)



def get_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Model (simple CNN for CIFAR‑10)
# ------------------------------

# class CIFARNet(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.25)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(256 * 4 * 4, 512)
#         self.fc2 = nn.Linear(512, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))  # 32->16
#         x = self.pool(F.relu(self.conv2(x)))  # 16->8
#         x = self.pool(F.relu(self.conv3(x)))  # 8->4
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x


# ------------------------------
# Data loading and partitioning
# ------------------------------

@dataclass
class ClientPartition:
    train_idx: np.ndarray




# def load_datasets(data_dir: str = "./data") -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
#     normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
#     train_tfms = T.Compose([
#         T.RandomCrop(32, padding=4),
#         T.RandomHorizontalFlip(),
#         T.ToTensor(),
#         normalize,
#     ])
#     test_tfms = T.Compose([T.ToTensor(), normalize])

#     trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
#     testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)
#     return trainset, testset


def iid_partitions(n_clients: int, n_samples: int) -> List[np.ndarray]:
    indices = np.random.permutation(n_samples)
    parts = np.array_split(indices, n_clients)
    return [p.astype(np.int64) for p in parts]


def shard_partitions(n_clients: int, y: np.ndarray, shards_per_client: int = 2) -> List[np.ndarray]:
    """Simple non-IID: sort by label, split into shards, assign shards per client."""
    n_shards = n_clients * shards_per_client
    idxs = np.arange(len(y))
    idxs_labels = np.vstack((idxs, y))
    # sort by label
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    shards = np.array_split(idxs_labels[0, :], n_shards)
    random.shuffle(shards)
    parts = [np.concatenate(shards[i * shards_per_client : (i + 1) * shards_per_client]).astype(np.int64) for i in range(n_clients)]
    return parts


# ------------------------------
# Flower Client
# ------------------------------

class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        model: nn.Module,
        device: torch.device,
        trainset: torchvision.datasets.CIFAR10,
        testloader: torch.utils.data.DataLoader,
        client_idx: np.ndarray,
        batch_size: int,
        lr: float,
        momentum: float,
    ) -> None:
        self.cid = cid
        self.model = model
        self.device = device
        self.trainset = trainset
        self.testloader = testloader
        self.client_idx = client_idx
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum

        self.model.to(self.device)

    # --- Flower NumPyClient interface ---
    def get_parameters(self, config: Dict[str, str] | None = None) -> List[np.ndarray]:
        return [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        state_keys = list(state_dict.keys())
        new_state = {k: torch.tensor(v) for k, v in zip(state_keys, parameters)}
        self.model.load_state_dict(new_state, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 3))
        batch_size = int(config.get("batch_size", self.batch_size))

        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(self.trainset, self.client_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(epochs):
            total = 0
            correct = 0
            running_loss = 0.0
            for (inputs, targets) in trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        metrics = {
            "train_loss": running_loss / max(1, total),
            "train_acc": correct / max(1, total),
        }
        return self.get_parameters({}), len(self.client_idx), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return test_loss / max(1, total), total, {"accuracy": correct / max(1, total)}


# ------------------------------
# Server-side evaluation function (centralized on test set)
# ------------------------------

def get_evaluate_fn(model: nn.Module, device: torch.device, testloader: torch.utils.data.DataLoader):
    def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict[str, str]):
        state_dict = model.state_dict()
        new_state = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
        model.load_state_dict(new_state, strict=True)
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                loss += batch_loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return loss / max(1, total), {"accuracy": correct / max(1, total)}

    return evaluate


# ------------------------------
# Simulation
# ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--fraction-fit", type=float, default=0.6)
    parser.add_argument("--partition", type=str, choices=["iid", "shards"], default="iid")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default=None, help='e.g. "cuda", "cpu", or leave empty to auto-detect')
    parser.add_argument("--use-ray", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)

    # Load datasets
    trainset, testset = load_datasets(args.data_dir)

    # Prepare test loader (centralized)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # Build client partitions
    if args.partition == "iid":
        parts = iid_partitions(args.clients, len(trainset))
    elif args.partition == "shards":
        targets = np.array(trainset.targets)
        parts = shard_partitions(args.clients, targets, shards_per_client=2)
    else:  # dominant
        targets = np.array(trainset.targets)
        parts = dominant_class_partitions(args.clients, targets, dominant_fraction=args.dominant_frac)

    # Global model template for server eval
    global_model = CIFARNet()

    # Strategy: FedAvg with 60% clients per round, ensure exactly the constraints
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,  # 0.6 of 100 => 60
        fraction_evaluate=0.0,  # we'll use centralized evaluate_fn instead
        min_fit_clients=math.ceil(args.clients * args.fraction_fit),
        min_available_clients=args.clients,
        evaluate_fn=get_evaluate_fn(global_model, device, testloader),
        on_fit_config_fn=lambda rnd: {
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
        },
    )

    # Client function factory
    def client_fn(cid: str) -> fl.client.Client:
        client_id_int = int(cid)
        model = CIFARNet()
        client = FlowerClient(
            cid=cid,
            model=model,
            device=device,
            trainset=trainset,
            testloader=testloader,  # each client can evaluate on the shared test set (optional)
            client_idx=parts[client_id_int],
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
        )
        return client

    client_resources = None
    if str(device) == "cuda":
        # Tell Flower/Ray each client needs a GPU if available
        client_resources = {"num_gpus": 1.0}

    if args.use_ray:
        # If Ray is installed, Flower will use it for parallelism automatically.
        os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")

    print(f"Starting simulation: {args.clients} clients, fraction_fit={args.fraction_fit}, rounds={args.rounds}")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )


if __name__ == "__main__":
    main()
