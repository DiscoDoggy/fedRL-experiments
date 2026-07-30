import torch
import torch.nn as nn

MAX_SEQ_LEN = 25
EMBED_DIM = 300
HIDDEN_DIM = 100


class Sent140LSTM(nn.Module):
    """2-layer LSTM for Sent140 binary sentiment classification.

    Architecture matches the q-FedAvg paper:
    - Embedding: vocab_size -> 300 (pre-trained GloVe, or learned)
    - 2-layer LSTM: 300 -> 100 hidden units
    - Linear: 100 -> num_classes (2)
    """

    def __init__(self, vocab_size, num_classes=2, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.embedding(x)
        # lstm_out: (batch, seq_len, hidden), (h_n, c_n)
        _, (h_n, _) = self.lstm(x)
        # Use last layer's final hidden state: h_n[-1]: (batch, hidden)
        out = self.classifier(h_n[-1])
        return out


def load_glove_embeddings(glove_path, vocab):
    """Load GloVe 300D vectors and create embedding matrix matching vocab.

    Args:
        glove_path: Path to glove.6B.300d.txt
        vocab: OrderedDict mapping word -> index

    Returns:
        numpy array of shape (vocab_size, 300), or None if file not found.
        Out-of-vocabulary words are initialized to zero.
    """
    import os
    import numpy as np

    if not os.path.exists(glove_path):
        return None

    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            glove[word] = vec

    vocab_size = len(vocab)
    embed_dim = 300
    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]

    return matrix
