#!/usr/bin/env python3
"""
Usage: python plot_accuracy_from_csv.py path/to/training_metrics.csv [--out output.pdf] [--label "My Run"]
Produces an accuracy vs round number line graph.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs round from a training_metrics CSV.")
    parser.add_argument("csv_path", help="Path to training_metrics.csv")
    parser.add_argument("--label", default="Accuracy", help="Legend label")
    parser.add_argument("--out", default=None, help="Output PDF path (default: same dir as CSV)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    if "round" not in df.columns or "accuracy" not in df.columns:
        raise ValueError("CSV must contain 'round' and 'accuracy' columns.")

    out_path = args.out or os.path.join(os.path.dirname(args.csv_path), "accuracy_vs_round.pdf")

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["round"], df["accuracy"], linewidth=2, label=args.label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Round")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"[OK] Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
