import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from torch.utils.data import Subset
from collections import Counter

def direchlet_partition(
    dataset,
    num_clients: int,
    num_classes: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    min_size_per_client: int = 0
):
    def _extract_labels(ds):
        if hasattr(ds, "targets"):
            return np.array(ds.targets)
        if hasattr(ds, "labels"):
            return np.array(ds.labels)
        return np.array([y for _, y in ds])

    rng = np.random.default_rng(seed)
    labels = _extract_labels(dataset)
    idxs = np.arange(len(labels))

    # indices per class, shuffled
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

        raw_counts = p * n
        counts = np.floor(raw_counts).astype(int)
        remainder = n - counts.sum()
        if remainder > 0:
            frac = raw_counts - counts
            top = np.argsort(-frac)[:remainder]
            counts[top] += 1

        start = 0
        for cid, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[cid].extend(c_idx[start:start+cnt].tolist())
                start += cnt

    if min_size_per_client > 0:
        changed = True
        while changed:
            changed = False
            donors = [i for i, lst in enumerate(client_indices) if len(lst) > min_size_per_client]
            takers = [i for i, lst in enumerate(client_indices) if len(lst) < min_size_per_client]
            if not donors or not takers:
                break
            di = 0
            for t in takers:
                need = min_size_per_client - len(client_indices[t])
                while need > 0 and di < len(donors):
                    d = donors[di]
                    give = max(0, len(client_indices[d]) - min_size_per_client)
                    if give > 0:
                        moved = client_indices[d][-min(need, give):]
                        client_indices[d] = client_indices[d][:-len(moved)]
                        client_indices[t].extend(moved)
                        need -= len(moved)
                        changed = True
                    else:
                        di += 1

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

    return [Subset(dataset, inds) for inds in client_indices]


def plot_stacked_client_class_distributions(client_datasets, num_classes=10, out_path='./client_class_dist.pdf', show=False, max_clients=None):
    """
    One bar per client (x-axis), stacked by class (0..num_classes-1) with legend.
    """
    n_clients = len(client_datasets) if max_clients is None else min(len(client_datasets), max_clients)

    base_ds = client_datasets[0].dataset
    fast_targets = None
    if hasattr(base_ds, "targets"):
        fast_targets = np.array(base_ds.targets)
    elif hasattr(base_ds, "labels"):
        fast_targets = np.array(base_ds.labels)

    counts_mat = np.zeros((n_clients, num_classes), dtype=int)
    for cid in range(n_clients):
        subset = client_datasets[cid]
        if fast_targets is not None and hasattr(subset, "indices"):
            labels = fast_targets[np.array(subset.indices)]
        else:
            labels = [subset.dataset[i][1] for i in subset.indices]
        cts = Counter(labels)
        for c in range(num_classes):
            counts_mat[cid, c] = cts.get(c, 0)

    xs = np.arange(n_clients)
    bottoms = np.zeros(n_clients, dtype=int)

    fig, ax = plt.subplots(figsize=(max(10, n_clients * 0.15), 6))

    bars = []
    for c in range(num_classes):
        bar = ax.bar(xs, counts_mat[:, c], bottom=bottoms, label=f"Class {c}")
        bars.append(bar)
        bottoms += counts_mat[:, c]

    ax.set_xlabel("Client")
    ax.set_ylabel("Samples")
    ax.set_title("Per-Client Class Distribution (Stacked)")
    ax.set_xticks(xs)
    if n_clients > 40:
        step = max(1, n_clients // 20)
        ax.set_xticks(xs[::step])
        ax.set_xticklabels([str(i) for i in xs[::step]])
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(ncol=min(5, num_classes), title="Class", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# def main():
#     transform_train_cifar = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     cifar_dataset = torchvision.datasets.CIFAR10(
#         root="./data", train=True, download=True, transform=transform_train_cifar
#     )

#     num_clients = 100
#     client_subsets = direchlet_partition(
#         dataset=cifar_dataset,
#         num_clients=num_clients,
#         num_classes=10,
#         alpha=0.1,    
#         seed=42,
#         min_size_per_client=0
#     )

#     plot_stacked_client_class_distributions(client_subsets, num_classes=10, out_path="./client_class_dist.pdf", show=True)

# # if __name__ == "__main__":
# #     main()

    

    
    
    

    

    

