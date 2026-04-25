#!/usr/bin/env python3
# Tall 2x2 scatterplot for single-column papers
# - Participation Frequency vs Processed Client Accuracies
# - Dataset-specific scaling (fonts + figure size) for CIFAR
# - Larger fonts/lines, 300 DPI PDF export
# - Per-subplot legends inside panels
# - Ablated vs Epsilon-0.5 comparison

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
BASE_DIR = "."
ABLATED_DIR = os.path.join(BASE_DIR, "results_for_runs_cifar_fairness_abalated")
EPSILON_DIR = os.path.join(BASE_DIR, "results_for_runs_cifar_fairness_epsilon_0.5")

DATASET_TAG = "cifar"   # <-- set to "cifar" when plotting CIFAR-10
CLIENT_COUNTS = [5, 10, 20, 30]

# ===================== DATASET-SPECIFIC SCALING =====================
if "cifar" in DATASET_TAG.lower():
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    FIGSIZE = (16.0, 4.0)   # horizontal layout for 4 subplots
else:
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    FIGSIZE = (14.0, 3.8)

# ===================== HELPERS =====================
def subdir_name(m: int) -> str:
    return f"100_clients_{m}_per_round_cifar"

def load_client_data(path: str):
    """Load participation_frequency and processed_client_accuracies"""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None, None
    
    participation = data.get("participation_frequency")
    accuracies = data.get("processed_client_accuracies")
    
    if not isinstance(participation, dict) or not isinstance(accuracies, dict):
        print(f"[WARN] Missing/invalid client data in {path}")
        return None, None
    
    return participation, accuracies

def load_results_for_m(m: int):
    """Load results for both ablated and epsilon versions"""
    sub = subdir_name(m)
    ablated_path = os.path.join(ABLATED_DIR, sub, "run_results.json")
    epsilon_path = os.path.join(EPSILON_DIR, sub, "run_results.json")

    if not os.path.isfile(ablated_path) or not os.path.isfile(epsilon_path):
        print(f"[WARN] Missing data for m={m}")
        return None

    ablated_part, ablated_acc = load_client_data(ablated_path)
    epsilon_part, epsilon_acc = load_client_data(epsilon_path)
    
    if ablated_part is None or epsilon_part is None:
        return None

    return {
        "m": m,
        "ablated": {"participation": ablated_part, "accuracies": ablated_acc},
        "epsilon": {"participation": epsilon_part, "accuracies": epsilon_acc},
    }

def _flatten_axes(axes):
    if isinstance(axes, np.ndarray):
        return axes.flatten().tolist()
    return [axes]

def extract_plot_data(participation_freq, processed_acc):
    """
    Extract x (participation) and y (accuracy) for scatter plot.
    Only include clients that exist in both dictionaries.
    """
    if not participation_freq or not processed_acc:
        return None, None
    
    # Convert string keys to integers for consistency
    part_dict = {}
    for k, v in participation_freq.items():
        try:
            part_dict[int(k)] = v
        except (ValueError, TypeError):
            continue
    
    acc_dict = {}
    for k, v in processed_acc.items():
        try:
            acc_dict[int(k)] = v
        except (ValueError, TypeError):
            continue
    
    # Find common clients
    common_clients = set(part_dict.keys()) & set(acc_dict.keys())
    if not common_clients:
        return None, None
    
    x = np.array([part_dict[c] for c in common_clients])
    y = np.array([acc_dict[c] for c in common_clients])
    
    return x, y

# ===================== PLOTTING (HORIZONTAL 1x4) =====================
def plot_participation_vs_accuracy_2x2(results_by_m, out_pdf: str):
    """
    Horizontal 1x4 figure for better linear correlation visualization.
    - Participation Frequency vs Client Accuracy Contribution
    - CIFAR: landscape orientation with larger font sizes
    - Ablated (circles) vs Epsilon-0.5 (squares)
    """
    is_cifar = "cifar" in DATASET_TAG.lower()

    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE, sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.3, wspace=0.35)
    axes = _flatten_axes(axes)

    colors = {
        "ablated": "#ff7f0e",    # orange
        "epsilon": "#1f77b4",   # blue
    }
    
    markers = {
        "ablated": "s",
        "epsilon": "o",
    }

    for i, m in enumerate(CLIENT_COUNTS):
        ax = axes[i]
        ax.set_title(rf"$|𝒮_t|$ = {m} clients/round")
        ax.grid(True, alpha=0.3)

        res = results_by_m.get(m)
        if not res:
            ax.text(0.5, 0.5, f"No data for m={m}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Extract data for ablated
        x_ablated, y_ablated = extract_plot_data(
            res["ablated"]["participation"],
            res["ablated"]["accuracies"]
        )
        
        # Extract data for epsilon
        x_epsilon, y_epsilon = extract_plot_data(
            res["epsilon"]["participation"],
            res["epsilon"]["accuracies"]
        )

        # Plot ablated
        if x_ablated is not None and len(x_ablated) > 0:
            ax.scatter(x_ablated, y_ablated, 
                      color=colors["ablated"], 
                      marker=markers["ablated"],
                      s=50, alpha=0.65, 
                      label="Ablated", edgecolors="black", linewidth=0.8)

        # Plot epsilon
        if x_epsilon is not None and len(x_epsilon) > 0:
            ax.scatter(x_epsilon, y_epsilon, 
                      color=colors["epsilon"], 
                      marker=markers["epsilon"],
                      s=50, alpha=0.65, 
                      label="FedRL", edgecolors="black", linewidth=0.8)

        # Axis labels
        ax.set_xlabel("Participation Frequency")
        ax.set_ylabel("Client Accuracy Contribution")

        # Per-subplot legend inside panel
        ax.legend(
            loc="best",
            fontsize=plt.rcParams["legend.fontsize"],
            frameon=False,
            labelspacing=0.25,
            borderpad=0.25,
            handlelength=1.5,
        )

    fig.suptitle("CIFAR-10: Client Participation vs Accuracy Contribution (Ablated vs FedRL)",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    print(f"[OK] Saved {out_pdf}")
    plt.close(fig)

# ===================== MAIN =====================
def main():
    results_by_m = {m: load_results_for_m(m) for m in CLIENT_COUNTS}
    
    # Check how many results loaded successfully
    valid_count = sum(1 for r in results_by_m.values() if r is not None)
    print(f"[INFO] Loaded results for {valid_count}/{len(CLIENT_COUNTS)} client counts")
    
    plot_participation_vs_accuracy_2x2(
        results_by_m,
        out_pdf="participation_vs_accuracy_ablated_vs_epsilon_2x2.pdf"
    )

if __name__ == "__main__":
    main()
