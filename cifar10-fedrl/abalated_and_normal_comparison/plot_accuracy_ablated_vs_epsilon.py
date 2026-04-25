#!/usr/bin/env python3
# Tall 2x2 plotting for single-column papers
# - Dataset-specific scaling (fonts + figure size) for CIFAR vs MNIST
# - Different accuracy y-axis start: MNIST=0.8, CIFAR=0.3
# - Larger fonts/lines, 300 DPI PDF export
# - Per-subplot legends inside panels
# - Optional CIFAR overlap aids (sparse markers + light gap shading)

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
EXPECTED_LEN = 200

# Enable extra visual aids to reduce line overlap on CIFAR accuracy plots
ENABLE_CIFAR_OVERLAP_AIDS = True

# ===================== DATASET-SPECIFIC SCALING =====================
if "cifar" in DATASET_TAG.lower():
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    FIGSIZE = (8.5, 10.5)   # taller, bigger for readability in a single column
else:
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    FIGSIZE = (7.0, 9.5)

# ===================== HELPERS =====================
def subdir_name(m: int) -> str:
    return f"100_clients_{m}_per_round_cifar"

def load_accuracies(path: str):
    """Load accuracies from run_results.json"""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None
    acc = data.get("accuracies")
    if not isinstance(acc, list):
        print(f"[WARN] Missing/invalid 'accuracies' in {path}")
        return None
    return acc

def load_results_for_m(m: int):
    """Load results for both ablated and epsilon versions"""
    sub = subdir_name(m)
    ablated_path = os.path.join(ABLATED_DIR, sub, "run_results.json")
    epsilon_path = os.path.join(EPSILON_DIR, sub, "run_results.json")

    if not os.path.isfile(ablated_path) or not os.path.isfile(epsilon_path):
        print(f"[WARN] Missing data for m={m}")
        return None

    ablated_acc = load_accuracies(ablated_path)
    epsilon_acc = load_accuracies(epsilon_path)
    
    if ablated_acc is None or epsilon_acc is None:
        return None

    return {
        "m": m,
        "ablated": ablated_acc,
        "epsilon": epsilon_acc,
    }

def _flatten_axes(axes):
    if isinstance(axes, np.ndarray):
        return axes.flatten().tolist()
    return [axes]

# ===================== PLOTTING (TALL 2x2) =====================
def plot_accuracy_comparison_2x2(results_by_m, out_pdf: str):
    """
    Tall 2x2 figure for single-column papers.
    - CIFAR: bigger fonts/figsize
    - Accuracy y-lims: CIFAR=0.1..0.85
    - Optional overlap aids on CIFAR accuracy plots (sparse markers + light gap shading)
    """
    is_cifar = "cifar" in DATASET_TAG.lower()
    acc_lower = 0.1 if is_cifar else 0.8

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.28, wspace=0.18)
    axes = _flatten_axes(axes)

    colors = {
        "ablated": "#ff7f0e",    # orange
        "epsilon": "#1f77b4",   # blue
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

        y_ablated = res["ablated"]
        y_epsilon = res["epsilon"]
        
        if not y_ablated or not y_epsilon:
            continue

        n = min(len(y_ablated), len(y_epsilon))
        x = np.arange(1, n + 1)

        # High-contrast styling; for CIFAR accuracy, use sparse markers to distinguish
        mark_every = max(1, n // 40) if (is_cifar and ENABLE_CIFAR_OVERLAP_AIDS) else None
        ablated_args = {"lw": 2.4}
        epsilon_args = {"lw": 2.4}
        if mark_every is not None:
            ablated_args.update({"marker": "o", "markevery": mark_every, "ms": 3.2})
            epsilon_args.update({"marker": "s", "markevery": mark_every, "ms": 3.0})

        ax.plot(x, y_ablated[:n], "--", color=colors["ablated"], 
               label="Ablated", **ablated_args)
        ax.plot(x, y_epsilon[:n], "-", color=colors["epsilon"], 
               label="FedRL", **epsilon_args)

        # Light gap shading to reveal subtle differences on CIFAR accuracy
        if is_cifar and ENABLE_CIFAR_OVERLAP_AIDS:
            ax.fill_between(x, y_ablated[:n], y_epsilon[:n], alpha=0.12, zorder=0)

        # Axis labels
        if i // 2 == 1:
            ax.set_xlabel("Federated Round")
        if i % 2 == 0:
            ax.set_ylabel("Accuracy")

        # Accuracy y-lims per dataset
        ax.set_ylim(acc_lower, 0.85)

        # Per-subplot legend inside panel
        ax.legend(
            loc="lower right",
            fontsize=plt.rcParams["legend.fontsize"],
            frameon=False,
            labelspacing=0.25,
            borderpad=0.25,
            handlelength=2.2,
        )

    fig.suptitle(f"CIFAR-10: Accuracy Progression (Ablated vs FedRL)",
                 y=0.915, fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    print(f"[OK] Saved {out_pdf}")
    plt.close(fig)

# ===================== MAIN =====================
def main():
    results_by_m = {m: load_results_for_m(m) for m in CLIENT_COUNTS}
    
    # Check how many results loaded successfully
    valid_count = sum(1 for r in results_by_m.values() if r is not None)
    print(f"[INFO] Loaded results for {valid_count}/{len(CLIENT_COUNTS)} client counts")
    
    plot_accuracy_comparison_2x2(
        results_by_m,
        out_pdf="accuracy_comparison_ablated_vs_epsilon_2x2.pdf"
    )

if __name__ == "__main__":
    main()
