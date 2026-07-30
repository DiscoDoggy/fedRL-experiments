import json
import re
import os
from collections import Counter, OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset


MAX_SEQ_LEN = 25


class Sent140Dataset(Dataset):
    """PyTorch Dataset for a single Sent140 user's tweets.

    Returns (token_indices, label) where token_indices is padded/truncated
    to MAX_SEQ_LEN (25) and label is 0 (negative) or 1 (positive).
    """

    def __init__(self, texts, labels, word2idx):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self._tokenize(self.texts[idx])
        indices = [self.word2idx.get(t, self.word2idx.get("<UNK>", 0)) for t in tokens]
        indices = indices[:MAX_SEQ_LEN]
        pad_len = MAX_SEQ_LEN - len(indices)
        indices = indices + [self.word2idx.get("<PAD>", 0)] * pad_len
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    @staticmethod
    def _tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    @staticmethod
    def build_vocab(all_texts, vocab_size=5000):
        counter = Counter()
        for text in all_texts:
            tokens = Sent140Dataset._tokenize(text)
            counter.update(tokens)
        most_common = counter.most_common(vocab_size)
        # Reserve 0=<PAD>, 1=<UNK>
        vocab = OrderedDict()
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
        for i, (word, _) in enumerate(most_common):
            vocab[word] = i + 2
        return vocab


def load_sent140(json_path, num_clients, min_samples=30, vocab_size=5000, seed=42):
    """Load Sent140 data and return (train_datasets, test_datasets, vocab).

    Each user with < min_samples is filtered out. Up to num_clients users are
    selected, sorted by sample count (largest first).
    Uses paper's 80/20 train/test split (no validation set needed for FL).
    """
    with open(json_path) as f:
        data = json.load(f)

    rng = np.random.default_rng(seed)

    user_ids = []
    user_texts = []
    user_labels = []
    for uid, ns in zip(data["users"], data["num_samples"]):
        if ns >= min_samples:
            x = data["user_data"][uid]["x"]
            y = data["user_data"][uid]["y"]
            # Tweet text is second-to-last element; last is split flag
            texts = [xi[-2] for xi in x]
            labels = [1 if yi == "4" else 0 for yi in y]
            user_texts.append(texts)
            user_labels.append(labels)

    # Sort by sample count descending, take top num_clients
    paired = list(zip(user_texts, user_labels))
    paired.sort(key=lambda p: len(p[0]), reverse=True)
    paired = paired[:num_clients]

    user_texts, user_labels = zip(*paired) if paired else ([], [])

    # Build vocabulary from all selected users' texts
    all_texts = []
    for texts in user_texts:
        all_texts.extend(texts)
    vocab = Sent140Dataset.build_vocab(all_texts, vocab_size)

    # 80/20 train/test split per user
    train_datasets = []
    test_datasets = []
    for texts, labels in zip(user_texts, user_labels):
        n = len(texts)
        n_test = max(1, int(n * 0.2))
        perm = rng.permutation(n)
        train_ds = Sent140Dataset(
            [texts[i] for i in perm[n_test:]],
            [labels[i] for i in perm[n_test:]],
            vocab,
        )
        test_ds = Sent140Dataset(
            [texts[i] for i in perm[:n_test]],
            [labels[i] for i in perm[:n_test]],
            vocab,
        )
        train_datasets.append(train_ds)
        test_datasets.append(test_ds)

    return train_datasets, test_datasets, vocab
