#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_client_data(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    participation = data.get("participation_frequency")
    accuracies = data.get("processed_client_accuracies")

    if not isinstance(participation, dict) or not isinstance(accuracies, dict):
        raise ValueError(f"Missing or invalid 'participation_frequency'/'processed_client_accuracies' in {path}")

    part_dict = {int(k): v for k, v in participation.items()}
    acc_dict = {int(k): v for k, v in accuracies.items()}

    common = sorted(set(part_dict) & set(acc_dict))
    if not common:
        raise ValueError("No common client keys between participation and accuracies.")

    x = np.array([part_dict[c] for c in common])
    y = np.array([acc_dict[c] for c in common])
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Plot participation frequency vs client accuracy contribution.")
    parser.add_argument("json_path", help="Path to run_results.json")
    parser.add_argument("--label", default="Run", help="Legend label for the scatter points")
    parser.add_argument("--out", default=None, help="Output PDF path (default: <json_dir>/participation_scatter.pdf)")
    args = parser.parse_args()

    x, y = load_client_data(args.json_path)

    out_path = args.out or os.path.join(os.path.dirname(args.json_path), "participation_scatter.pdf")

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

    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + b, "--", linewidth=1.5, color="gray", label=f"Trend (r={np.corrcoef(x, y)[0,1]:.2f})")

    ax.set_xlabel("Participation Frequency")
    ax.set_ylabel("Client Accuracy Contribution")
    ax.set_title(f"Participation vs Accuracy Contribution\n{os.path.basename(args.json_path)}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"[OK] Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
