#!/bin/bash
# Download and preprocess F-EMNIST from the LEAF benchmark.
# Run once from the project root before submitting any femnist jobs.
#
#   bash src/download_femnist.sh
#
# Requires: git, python3, numpy, Pillow (pip install numpy pillow)
# Output:   data/femnist/train/*.json   data/femnist/test/*.json
#           (one JSON file per 100 writers, ~35 files each)

set -e
cd "$(dirname "$0")/.."   # project root

DEST="data/femnist"
LEAF_DIR="data/leaf_femnist_repo"

echo "==> Cloning LEAF repository (shallow)..."
if [ ! -d "$LEAF_DIR" ]; then
    git clone --depth 1 https://github.com/TalwalkarLab/leaf.git "$LEAF_DIR"
fi

echo "==> Generating F-EMNIST data (this takes 10-30 min)..."
cd "$LEAF_DIR/data/femnist"

# --sf 1.0  = use 100% of writers (~3,550)
# --t sample  = use standard LEAF train/test split
# --smplseed / --spltseed  = reproducible sampling & splitting
bash preprocess.sh -s niid --sf 1.0 --k 0 --t sample \
    --smplseed 0 --spltseed 0

cd "$(dirname "$0")/.."   # back to project root

echo "==> Copying JSON files to $DEST/..."
mkdir -p "$DEST/train" "$DEST/test"
cp "$LEAF_DIR/data/femnist/data/train/"*.json "$DEST/train/"
cp "$LEAF_DIR/data/femnist/data/test/"*.json  "$DEST/test/"

echo "==> Done. Train files: $(ls $DEST/train/*.json | wc -l)  Test files: $(ls $DEST/test/*.json | wc -l)"
echo "    You can now run: sbatch src/run_femnist.sh"
