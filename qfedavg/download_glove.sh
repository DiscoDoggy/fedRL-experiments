#!/bin/bash
# Download GloVe 6B 300D embeddings for Sent140 LSTM model.
# Usage: bash download_glove.sh

set -e

GLOVE_DIR="/Users/924322786/programming/Untitled Folder/fedrl-combined/data/glove"
GLOVE_URL="https://nlp.stanford.edu/data/glove.6B.zip"

mkdir -p "$GLOVE_DIR"

if [ -f "$GLOVE_DIR/glove.6B.300d.txt" ]; then
    echo "GloVe 300D already exists at $GLOVE_DIR"
    exit 0
fi

echo "Downloading GloVe 6B (822MB)..."
cd "$GLOVE_DIR"
curl -OL "$GLOVE_URL"

echo "Extracting..."
unzip -o glove.6B.zip

echo "Cleaning up..."
rm glove.6B.zip

echo "Done. GloVe embeddings saved to $GLOVE_DIR"
ls -la "$GLOVE_DIR"/glove.6B.*.txt
