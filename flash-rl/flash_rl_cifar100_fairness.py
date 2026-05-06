"""
FLASH-RL CIFAR-100 Fairness Experiment
=======================================
Runs FLASH-RL (reputation-DQL client selection) on CIFAR-100 with Dirichlet
non-IID partitioning (α=0.5) and tracks per-client local accuracy every round.

Uses the same ResNetFed model and Dirichlet partitioner as the fedrl-combined
fairness experiment so results are directly comparable.

Loop: k ∈ [5, 10, 20, 30] clients per round, all saved under:
  fairness_results_flash_rl/flash_rl/k_<N>/fairness_results.json
"""

import os
import sys
import copy
import json
import pickle
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# ---- FLASH-RL internals (unchanged) ----
import serverFL.Server_FLASHRL as Server_FLASHRL
import clientFL.Client as ClientModule

# ==============================================================================
# Config
# ==============================================================================
NUM_CLIENTS        = 100
NUM_CLASSES        = 100
NUM_ROUNDS         = 200
CLIENTS_PER_ROUND_LIST = [5, 10, 20, 30]
LOCAL_EPOCHS       = 3
DIRICHLET_ALPHA    = 0.5
TEST_SPLIT_RATIO   = 0.2
BATCH_SIZE         = 32
RESULTS_BASE       = "fairness_results_flash_rl"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


# ==============================================================================
# ResNetFed — same architecture used in fairness_experiment.py
# ==============================================================================
class ResNetFed(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                     padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ==============================================================================
# Dirichlet partitioner — same logic as fedrl-combined/dirchlet_partitioner.py
# ==============================================================================
def dirichlet_partition(dataset, num_clients, num_classes=100, alpha=0.5,
                         seed=42, min_size_per_client=20):
    def _labels(ds):
        if hasattr(ds, "targets"):
            return np.array(ds.targets)
        return np.array([y for _, y in ds])

    rng = np.random.default_rng(seed)
    labels = _labels(dataset)
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
        remainder = n - counts.sum()
        if remainder > 0:
            frac = p * n - counts
            counts[np.argsort(-frac)[:remainder]] += 1
        start = 0
        for cid, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[cid].extend(c_idx[start:start + cnt].tolist())
                start += cnt

    # Ensure min size
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


def split_client_datasets(subsets, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    splits = []
    for subset in subsets:
        n = len(subset)
        n_test = max(1, int(n * test_ratio))
        perm = rng.permutation(n)
        test_sub  = Subset(subset, perm[:n_test].tolist())
        train_sub = Subset(subset, perm[n_test:].tolist())
        splits.append((train_sub, test_sub))
    return splits


# ==============================================================================
# Per-client local evaluator (wraps FLASH-RL Client)
# ==============================================================================
class FairnessClientWrapper:
    """Thin wrapper that adds local test evaluation on top of FLASH-RL Client."""

    def __init__(self, client_id, flash_client, test_subset, device):
        self.client_id   = client_id
        self.client      = flash_client
        self.test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
        self.device      = device
        self._eval_model = None

    def evaluate_local(self, global_state_dict):
        if self._eval_model is None:
            self._eval_model = ResNetFed(num_classes=NUM_CLASSES).to(self.device)
        self._eval_model.load_state_dict(global_state_dict)
        self._eval_model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                _, predicted = torch.max(self._eval_model(images), 1)
                correct += (predicted == labels).sum().item()
                total   += labels.size(0)
        return correct / total if total > 0 else 0.0


# ==============================================================================
# Main experiment
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None,
                        help="Single k value to run (default: all k in list)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- CIFAR-100 transforms ----
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    logger.info("Loading CIFAR-100...")
    train_ds = datasets.CIFAR100("data/cifar100/", train=True,
                                  download=True, transform=trans_train)
    test_ds  = datasets.CIFAR100("data/cifar100/", train=False,
                                  download=True, transform=trans_test)

    global_test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # ---- Dirichlet partition ----
    logger.info(f"Partitioning {NUM_CLIENTS} clients, α={DIRICHLET_ALPHA}...")
    client_subsets = dirichlet_partition(train_ds, NUM_CLIENTS, NUM_CLASSES,
                                          DIRICHLET_ALPHA, seed=42,
                                          min_size_per_client=20)
    splits = split_client_datasets(client_subsets, TEST_SPLIT_RATIO)
    logger.info(f"Partition done. Client sizes: "
                f"min={min(len(s[0]) for s in splits)}, "
                f"max={max(len(s[0]) for s in splits)}")

    # ---- Fake clients_info (FLASH-RL needs cores/freq/bw) ----
    # We use constant values so latency is uniform — the repo only uses it for
    # the reputation penalty, which we keep as-is.
    rng = np.random.default_rng(0)
    clients_info = [
        (f"client_{i}",                      # name
         len(splits[i][0]),                  # num samples
         rng.integers(2, 8).item(),          # cores
         [rng.uniform(1.0, 3.0)],            # frequency GHz (list)
         [rng.uniform(10.0, 100.0)])         # bandwidth Mbps (list)
        for i in range(NUM_CLIENTS)
    ]

    k_values = [args.k] if args.k is not None else CLIENTS_PER_ROUND_LIST

    for k in k_values:
        logger.info(f"\n{'#'*60}\n  k = {k}\n{'#'*60}")

        # Build dict_clients expected by FLASH-RL Server
        # (maps client name -> Subset of training data)
        dict_clients = {
            f"client_{i}": splits[i][0]
            for i in range(NUM_CLIENTS)
        }

        # Fresh global model
        global_model = ResNetFed(num_classes=NUM_CLASSES).to(device)

        # FLASH-RL server
        server = Server_FLASHRL.Server_FLASHRL(
            num_clients   = NUM_CLIENTS,
            global_model  = global_model,
            dict_clients  = dict_clients,
            loss_fct      = nn.CrossEntropyLoss(),
            B             = BATCH_SIZE,
            dataset_test  = test_ds,
            learning_rate = 0.001,
            momentum      = 0.9,
            clients_info  = clients_info,
            device        = device,
        )

        # Build per-client fairness wrappers for local eval
        fairness_clients = [
            FairnessClientWrapper(i, server.list_clients[i],
                                   splits[i][1], device)
            for i in range(NUM_CLIENTS)
        ]

        # Run FLASH-RL training with per-client eval hook
        C = k / NUM_CLIENTS
        results = _run_with_local_eval(
            server, fairness_clients, global_test_loader,
            k, C, device
        )

        # Save
        out_dir = os.path.join(RESULTS_BASE, "flash_rl", f"k_{k}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "fairness_results.json")
        serialisable = {
            "clients_per_round": k,
            "global_accuracies": results["global_accuracies"],
            "participation_frequency": {
                str(cid): v
                for cid, v in results["participation_frequency"].items()
            },
            "per_client_accuracies": [
                {str(cid): acc for cid, acc in rd.items()}
                for rd in results["per_client_accuracies"]
            ],
        }
        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"Saved → {out_path}")

    logger.info("\nAll k values complete.")


def _run_with_local_eval(server, fairness_clients, global_test_loader,
                          k, C, device):
    """
    Run the FLASH-RL training loop manually so we can inject per-client
    local evaluation each round. Mirrors global_train_others() logic but
    calls evaluate_local() on all 100 clients after each aggregation.
    """
    import copy as _copy
    from sklearn.decomposition import PCA

    num_clients = server.N
    m = int(max(C * num_clients, 1))
    mu = 0
    lamb = 0.6
    rep_init = 1 / num_clients
    batch_size = 32
    E = LOCAL_EPOCHS
    type_data = "others"

    # Tracking
    global_accuracies    = []
    per_client_accuracies = []
    participation_freq   = {}
    rewards_all          = []
    reputation_clients_t = np.full(num_clients, rep_init)

    num_param = sum(p.numel() for p in server.model.parameters()
                    if p.requires_grad)

    # ---- Build initial state (same as Server_FLASHRL) ----
    weight_list  = []
    nsamples     = []
    ncores       = []
    freqs        = []
    bws          = []

    max_latency = 0
    min_latency = 1e9

    for client in server.list_clients:
        freq_c = random.choice(client.frequency)
        bw_c   = random.choice(client.bandwidth)
        lat_min = (client.get_size() * 64 * 40 * 20) / (
            client.numbercores * 1e6 * max(client.frequency)
        ) + (num_param * 64) / (1e6 * max(client.bandwidth))
        lat_max = (client.get_size() * 64 * 40 * 20) / (
            client.numbercores * 1e6 * min(client.frequency)
        ) + (num_param * 64) / (1e6 * min(client.bandwidth))
        max_latency = max(max_latency, lat_max)
        min_latency = min(min_latency, lat_min)

        w = client.train(server.model.state_dict(), 1, mu, type_data)
        weight_list.append(server.flatten(w))
        nsamples.append(client.get_size())
        ncores.append(client.numbercores)
        freqs.append(freq_c)
        bws.append(bw_c)

    pca = PCA(n_components=num_clients)
    wl_pca = pca.fit_transform(weight_list)

    state_list = []
    for i in range(num_clients):
        state_list.append([
            list(wl_pca[i]), nsamples[i], ncores[i], freqs[i], bws[i]
        ])

    state = server.flatten_state(state_list)

    from RL import DQL
    dql = DQL.DQL(len(state), num_clients, batch_size)

    prev_acc, _ = server.test(type_data)

    # ---- Main loop ----
    for comm_round in range(NUM_ROUNDS):
        logger.info(f"  [FLASH-RL k={k}] Round {comm_round+1}/{NUM_ROUNDS}")

        global_weights = server.model.state_dict()

        if (comm_round + 1) % dql.update_rate == 0:
            dql.update_target_network()

        # Client selection
        if comm_round == 0:
            active_idx = server.select_active_clients_random(comm_round, C)
        else:
            active_idx = dql.multiaction_selection(state, C, comm_round,
                                                    mode="Mode1")
        active_idx.sort()

        # Track participation
        for cid in active_idx:
            participation_freq[cid] = participation_freq.get(cid, 0) + 1

        # Local training + weighted aggregation
        active_clients = [server.list_clients[i] for i in active_idx]
        scaled_weights = []
        local_flat     = []
        time_roundt    = []

        for cid in active_idx:
            client = server.list_clients[cid]
            w = client.train(global_weights, E, mu, type_data)
            local_flat.append(server.flatten(w))

            state_list[cid][0] = list(
                (pca.transform(
                    np.array(server.flatten(_copy.deepcopy(w))).reshape(1, -1)
                ))[0]
            )
            sf = server.weight_scalling_factor(client, active_clients)
            scaled_weights.append(server.scale_model_weights(w, sf))

            freq_c = random.choice(client.frequency)
            bw_c   = random.choice(client.bandwidth)
            latency = (client.get_size() * 64 * 40 * 20) / (
                client.numbercores * 1e6 * freq_c
            ) + (num_param * 64) / (1e6 * bw_c)
            state_list[cid][3] = freq_c
            state_list[cid][4] = bw_c
            time_roundt.append(latency)

        avg_w = server.sum_scaled_weights(scaled_weights)
        server.model.load_state_dict(avg_w)

        # Global accuracy
        curr_acc, loss_test = server.test(type_data)
        curr_acc_val = curr_acc.cpu().item() if isinstance(curr_acc, torch.Tensor) else curr_acc
        global_accuracies.append(curr_acc_val)
        logger.info(f"     Global acc: {curr_acc_val:.4f}  loss: {loss_test:.4f}")

        # Per-client local evaluation
        gsd = server.model.state_dict()
        round_client_accs = {}
        for fc in fairness_clients:
            round_client_accs[fc.client_id] = fc.evaluate_local(gsd)
        per_client_accuracies.append(round_client_accs)

        # Reward / reputation (same as Server_FLASHRL)
        next_state = server.flatten_state(state_list)
        action = np.zeros(num_clients)
        action[active_idx] = 1

        flat_avg = np.array(server.flatten(server.model.state_dict()))
        nd = 1 / num_param * np.sum(
            (np.array(local_flat) - flat_avg) / (flat_avg + 1e-9), axis=1
        )
        prev_acc_val = prev_acc.cpu().item() if isinstance(prev_acc, torch.Tensor) else prev_acc
        if curr_acc_val > prev_acc_val:
            utility = np.exp(-np.abs(nd))
        else:
            utility = 1 - np.exp(-np.abs(nd))

        lat_range = max_latency - min_latency
        if lat_range > 0:
            norm_lat = (np.array(time_roundt) - min_latency) / lat_range
        else:
            norm_lat = np.zeros(len(time_roundt))

        reputation_clients_t[active_idx] = (
            (1 - lamb) * reputation_clients_t[active_idx]
            + lamb * (utility - norm_lat)
        )
        reward = reputation_clients_t[active_idx].copy()
        rewards_all.append(reward)

        done = (comm_round == NUM_ROUNDS - 1)
        dql.store_transistion(state, action, reward, next_state, done)
        state = _copy.deepcopy(next_state)
        prev_acc = curr_acc

        dql.train(comm_round, mode="Mode1")

    return {
        "global_accuracies":     global_accuracies,
        "per_client_accuracies": per_client_accuracies,
        "participation_frequency": participation_freq,
    }


# ==============================================================================
# Client.get_size() helper — FLASH-RL Client may not have it already
# ==============================================================================
def _patch_client():
    if not hasattr(ClientModule.Client, "get_size"):
        def _get_size(self):
            return len(self.local_data.dataset)
        ClientModule.Client.get_size = _get_size

    if not hasattr(ClientModule.Client, "get_model"):
        def _get_model(self):
            return _copy.deepcopy(self.local_model.state_dict())
        ClientModule.Client.get_model = _get_model


if __name__ == "__main__":
    import copy as _copy
    _patch_client()
    main()
