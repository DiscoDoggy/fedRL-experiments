"""F-EMNIST per-writer dataset loader for the LEAF benchmark.

Data source: https://github.com/TalwalkarLab/leaf/tree/master/data/femnist

Expected directory layout (relative to project root):
    data/femnist/train/*.json   – LEAF train JSON files
    data/femnist/test/*.json    – LEAF test JSON files

Each JSON file has the structure:
    {
        "users":       ["f0000_14", ...],
        "num_samples": [n0, n1, ...],
        "user_data": {
            "f0000_14": {
                "x": [[...784 floats...], ...],   # pixel values in [0, 1]
                "y": [3, 5, ...]                  # class labels 0-61
            },
            ...
        }
    }

Classes: 62  (0-9 digits, A-Z uppercase, a-z lowercase)
Images:  28x28 grayscale, pre-flattened to 784 floats in [0, 1]
"""

import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 62  # 10 digits + 26 uppercase + 26 lowercase


class WriterDataset(Dataset):
    """PyTorch Dataset wrapping one writer's samples from LEAF F-EMNIST."""

    def __init__(self, xs: List[List[float]], ys: List[int]):
        self.x = torch.tensor(xs, dtype=torch.float32)   # (N, 784)
        self.y = torch.tensor(ys, dtype=torch.long)       # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return (1, 28, 28) tensor so it works with Conv2d if needed,
        # or (784,) for FC — both shapes are handled by FedFCNet.forward().
        return self.x[idx], self.y[idx]


def _load_json_dir(directory: str):
    """Load all LEAF JSON files from a directory, merge into one dict."""
    all_users, all_num_samples, all_user_data = [], [], {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(directory, fname)) as fh:
            d = json.load(fh)
        all_users.extend(d["users"])
        all_num_samples.extend(d["num_samples"])
        all_user_data.update(d["user_data"])
    return all_users, all_num_samples, all_user_data


def load_femnist_writers(
    data_dir: str = "./data/femnist",
    min_samples: int = 50,
    num_clients: int = 100,
    seed: int = 42,
) -> Tuple[List[str], List[WriterDataset], List[WriterDataset]]:
    """Load F-EMNIST, filter writers by minimum samples, select FL clients.

    Parameters
    ----------
    data_dir : str
        Root directory containing ``train/`` and ``test/`` subdirectories.
    min_samples : int
        Writers with fewer training samples than this threshold are excluded.
        Ensures each client has sufficient data to train meaningfully.
    num_clients : int
        How many writers to select as FL clients from the qualified pool.
    seed : int
        Random seed — controls which writers are selected, making the
        client population fully reproducible across runs.

    Returns
    -------
    writer_ids      : list[str]           — selected writer identifiers
    train_datasets  : list[WriterDataset] — one per selected writer
    test_datasets   : list[WriterDataset] — one per selected writer
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"F-EMNIST data not found at '{data_dir}'.\n"
            "Run  src/download_femnist.sh  from the project root first."
        )

    train_users, train_num_samples, train_data = _load_json_dir(train_dir)
    _,           _,                 test_data  = _load_json_dir(test_dir)

    # Filter: only writers with enough training samples that also appear
    # in the test set.
    qualified = [
        u for u, n in zip(train_users, train_num_samples)
        if n >= min_samples and u in test_data
    ]

    if len(qualified) < num_clients:
        raise ValueError(
            f"Only {len(qualified)} writers have >= {min_samples} training "
            f"samples, but {num_clients} clients were requested. "
            f"Lower min_samples or reduce num_clients."
        )

    # Reproducible selection — different seeds → different client pools.
    rng = random.Random(seed)
    selected = rng.sample(qualified, num_clients)

    writer_ids, train_datasets, test_datasets = [], [], []
    for uid in selected:
        writer_ids.append(uid)
        train_datasets.append(WriterDataset(
            train_data[uid]["x"], train_data[uid]["y"]
        ))
        test_datasets.append(WriterDataset(
            test_data[uid]["x"], test_data[uid]["y"]
        ))

    return writer_ids, train_datasets, test_datasets


def compute_global_class_dist(
    train_datasets: List[WriterDataset],
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Compute normalised global class distribution over all writers."""
    counts = np.zeros(num_classes)
    for ds in train_datasets:
        for lbl in ds.y.numpy():
            counts[int(lbl)] += 1
    return counts / counts.sum()
