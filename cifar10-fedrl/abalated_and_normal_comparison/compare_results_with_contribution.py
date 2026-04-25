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

print("HELLOW WORLD")

# ===================== CONFIG =====================
BASE_DIR   = "."
FEDAVG_DIR = os.path.join(BASE_DIR, "results_for_runs_fedavg")
FEDRL_DIR  = os.path.join(BASE_DIR, "results_for_runs_fedrl")

DATASET_TAG = "cifar"   # <-- set to "cifar" when plotting CIFAR-10
EXPECTED_LEN = 200
ALL_M = [10, 20, 30 , 40]

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
    return f"100_clients_{m}_per_round_{DATASET_TAG}"

def load_run_json(path: str):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None, None
    acc = data.get("accuracies")
    loss = data.get("losses")
    if not isinstance(acc, list) or not isinstance(loss, list):
        print(f"[WARN] Missing/invalid 'accuracies'/'losses' in {path}")
        return None, None
    return acc, loss

def load_results_for_m(m: int):
    sub = subdir_name(m)
    fedavg_path = os.path.join(FEDAVG_DIR, sub, "run_results.json")
    fedrl_path  = os.path.join(FEDRL_DIR,  sub, "run_results.json")

    if not os.path.isfile(fedavg_path) or not os.path.isfile(fedrl_path):
        print(f"[WARN] Missing data for |S_t|={m}")
        return None

    fa_acc, fa_loss = load_run_json(fedavg_path)
    fr_acc, fr_loss = load_run_json(fedrl_path)
    if fa_acc is None or fr_acc is None:
        return None

    return {
        "m": m,
        "fedavg": {"acc": fa_acc, "loss": fa_loss},
        "fedrl":  {"acc": fr_acc, "loss": fr_loss},
    }

def _flatten_axes(axes):
    if isinstance(axes, np.ndarray):
        return axes.flatten().tolist()
    return [axes]

# ===================== PLOTTING (TALL 2x2) =====================
def plot_grouped_2x2_tall(results_by_m, metric: str, out_pdf: str, title_prefix: str):
    """
    Tall 2x2 figure for single-column papers.
    Groups:
      (a) [5,10]   (b) [20,30]
      (c) [40,50]  (d) [60]
    - CIFAR: bigger fonts/figsize (set above)
    - Accuracy y-lims: MNIST=0.8..1.0, CIFAR=0.3..1.0
    - Optional overlap aids on CIFAR accuracy plots (sparse markers + light gap shading)
    """
    k_groups = [
        [10],
        [20],
        [30],
        [40],
    ]

    is_cifar = "cifar" in DATASET_TAG.lower()
    acc_lower = 0.1 if is_cifar else 0.8

    if is_cifar:
        title_prefix = 'CIFAR-10' + title_prefix[5:]

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.28, wspace=0.18)
    axes = _flatten_axes(axes)

    for i, group in enumerate(k_groups):
        ax = axes[i]
        ax.set_title(rf"$|𝒮_t|$ = {group}")
        ax.grid(True, alpha=0.3)

        handles, labels = [], []

        for m in group:
            res = results_by_m.get(m)
            if not res:
                continue
            y_fedrl  = res["fedrl"].get(metric, [])
            y_fedavg = res["fedavg"].get(metric, [])
            if not y_fedrl or not y_fedavg:
                continue

            n = min(len(y_fedrl), len(y_fedavg))
            x = np.arange(1, n + 1)

            # High-contrast styling; for CIFAR accuracy, use sparse markers to distinguish
            mark_every = max(1, n // 40) if (is_cifar and metric == "acc" and ENABLE_CIFAR_OVERLAP_AIDS) else None
            rl_args = {"lw": 2.4}
            av_args = {"lw": 2.4}
            if mark_every is not None:
                rl_args.update({"marker": "o", "markevery": mark_every, "ms": 3.2})
                av_args.update({"marker": "s", "markevery": mark_every, "ms": 3.0})

            h1, = ax.plot(x, y_fedrl[:n],  "-",  label=rf"FedRL ($|𝒮_t|={m}$)", **rl_args)
            h2, = ax.plot(x, y_fedavg[:n], "--", label=rf"FedAvg ($|𝒮_t|={m}$)", **av_args)
            handles.extend([h1, h2])
            labels.extend([rf"FedRL ($|𝒮_t|={m}$)", rf"FedAvg ($|𝒮_t|={m}$)"])

            # Light gap shading to reveal subtle differences on CIFAR accuracy
            if is_cifar and metric == "acc" and ENABLE_CIFAR_OVERLAP_AIDS:
                ax.fill_between(x, y_fedrl[:n], y_fedavg[:n], alpha=0.12, zorder=0)

        # Axis labels
        if i // 2 == 1:
            ax.set_xlabel("Federated Round")
        if i % 2 == 0:
            ax.set_ylabel("Accuracy" if metric == "acc" else "Loss")

        # Accuracy y-lims per dataset
        if metric == "acc":
            ax.set_ylim(acc_lower, 0.85)

        # Per-subplot legend inside panel
        # Per-subplot legend inside panel
        if handles:
            # Move legend for the top-left subplot (index 0) to top-left for CIFAR
            if is_cifar and i == 0:
                legend_loc = "upper left"
            else:
                legend_loc = "lower right"
            ax.legend(
                handles, labels,
                loc=legend_loc,
                fontsize=plt.rcParams["legend.fontsize"],
                frameon=False,
                labelspacing=0.25,
                borderpad=0.25,
                handlelength=2.2,
            )


    fig.suptitle(f"{title_prefix}: {'Accuracy vs Rounds' if metric=='acc' else 'Loss'}",
                 y=0.915 if is_cifar else 0.995, fontsize=17 if is_cifar else 15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_pdf}")

# ===================== MAIN =====================
def main():
    results_by_m = {m: load_results_for_m(m) for m in ALL_M}
    title_ds = DATASET_TAG.upper()

    # Accuracy (tall 2x2)
    plot_grouped_2x2_tall(
        results_by_m,
        metric="acc",
        out_pdf=f"{DATASET_TAG}_grouped_accuracy_tall_2x2.pdf",
        title_prefix=rf"{title_ds} (for varying $|𝒮_t|$)"
    )

    # Loss (tall 2x2)
    plot_grouped_2x2_tall(
        results_by_m,
        metric="loss",
        out_pdf=f"{DATASET_TAG}_grouped_loss_tall_2x2.pdf",
        title_prefix=rf"{title_ds} (for varying $|𝒮_t|$)"
    )

if __name__ == "__main__":
    main()
