#!/usr/bin/env python3
# Enlarged (taller) 2x2 plotting for single-column papers
# - Bigger vertical size (more readable in column layout)
# - Larger fonts/lines
# - Accuracy y-axis starts at 0.7
# - Per-subplot legends inside panels

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
BASE_DIR   = "."
FEDAVG_DIR = os.path.join(BASE_DIR, "results_for_runs_fedavg_cnn")
FEDRL_DIR  = os.path.join(BASE_DIR, "results_for_runs_fedrl_cnn")

DATASET_TAG = "mnist"   # <-- change to "cifar" when plotting CIFAR-10
EXPECTED_LEN = 200
ALL_M = [10,20,30,40]

# Readability: bigger fonts for column-width figures
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

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
    Tall 2x2 figure (better vertical readability in single-column papers).
    Groups:
      (a) [5,10]   (b) [20,30]
      (c) [40,50]  (d) [60]
    """
    k_groups = [
        [10],
        [20],
        [30],
        [40],
    ]

    # Taller than wide: better use of vertical space in a column
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 9.5), sharex=True, sharey=True)
    # Add a bit more vertical spacing between rows
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
            x = range(1, n + 1)
            # Thicker lines for visibility
            h1, = ax.plot(x, y_fedrl[:n],  '-',  lw=2.2, label=rf"FedRL ($|𝒮_t|={m}$)")
            h2, = ax.plot(x, y_fedavg[:n], '--', lw=2.2, label=rf"FedAvg ($|𝒮_t|={m}$)")
            handles.extend([h1, h2])
            labels.extend([rf"FedRL ($|𝒮_t|={m}$)", rf"FedAvg ($|𝒮_t|={m}$)"])

        # Axis labels
        if i // 2 == 1:
            ax.set_xlabel("Federated Round")
        if i % 2 == 0:
            ax.set_ylabel("Accuracy" if metric == "acc" else "Loss")

        # Focus accuracy view
        if metric == "acc":
            ax.set_ylim(0.9, 1.0)

        # Per-subplot legend inside panel (bottom-right)
        if handles:
            ax.legend(handles, labels,
                      loc="lower right", fontsize=11, frameon=False,
                      labelspacing=0.25, borderpad=0.25, handlelength=2.0)

    fig.suptitle(f"{title_prefix}: {'Accuracy vs Rounds' if metric=='acc' else 'Loss'}",
                 y=0.915, fontsize=15)
    # Tight layout but leave a little room for suptitle
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
