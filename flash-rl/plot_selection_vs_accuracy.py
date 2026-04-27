#!/usr/bin/env python3
"""
Usage: python plot_selection_vs_accuracy.py path/to/client_selection_summary.csv [--label "My Run"] [--out output.pdf]
Produces a scatter plot of selection count vs average accuracy impact per client.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot client selection frequency vs avg accuracy impact.")
    parser.add_argument("csv_path", help="Path to client_selection_summary.csv")
    parser.add_argument("--label", default="Clients", help="Legend label for scatter points")
    parser.add_argument("--out", default=None, help="Output PDF path (default: same dir as CSV)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    if "selection_count" not in df.columns or "avg_acc_impact" not in df.columns:
        raise ValueError("CSV must contain 'selection_count' and 'avg_acc_impact' columns.")

    out_path = args.out or os.path.join(os.path.dirname(args.csv_path), "selection_vs_accuracy.pdf")

    x = df["selection_count"].values
    y = df["avg_acc_impact"].values

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=55, alpha=0.7, edgecolors="black", linewidth=0.8, label=args.label)

    ax.set_xlabel("Selection Frequency")
    ax.set_ylabel("Avg Accuracy Impact")
    ax.set_title(f"Client Selection Frequency vs Avg Accuracy Impact")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"[OK] Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
