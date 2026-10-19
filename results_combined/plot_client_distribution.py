#!/usr/bin/env python3
"""Plot client-accuracy distributions from results_combined/.

Produces 4 variants so the user can pick the best look:
  A) 2x4 grid  (rows=CIFAR-10/CIFAR-100, cols=k=5,10,20,30)
  B) 2x2 grid  (CIFAR-10 k=5,10  ;  CIFAR-100 k=20,30)
Each in both violin and 1D-jittered-scatter styles.

Methods (ablation intentionally excluded): FedRL(kl_capped), FAVOR, FedAvg, QFedAvg.
Data: final-round per-client accuracies (per_client_accuracies[-1]).
"""

import os
import json
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")

METHODS = [
    ("fedrl_kl_capped", "FedRL", "#1f77b4"),
    ("favor", "FAVOR", "#d62728"),
    ("fedavg", "FedAvg", "#2ca02c"),
    ("qfedavg", "q-FedAvg", "#ff7f0e"),
]

# Option A: full 2x4 grid
OPTION_A = {
    "cifar10": [5, 10, 20, 30],
    "cifar100": [5, 10, 20, 30],
}
# Option B: strongest-k selection
OPTION_B = {
    "cifar10": [5, 10],
    "cifar100": [20, 30],
}

DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}


def load_final_per_client(dataset, method, k):
    path = os.path.join(RESULTS_ROOT, dataset, method, f"k{k}", "run_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    pc = d.get("per_client_accuracies") or d.get("final_per_client_accuracies")
    if not pc:
        return None
    last = pc[-1]
    if isinstance(last, dict):  # favor stores final as dict
        last = [last[str(i)] for i in range(len(last))]
    vals = [float(a) for a in last if a is not None]
    return vals if len(vals) > 1 else None


def make_violin_ax(ax, data_by_method, horizontal=False):
    """data_by_method: list of (label, values). Draws violins + median marker."""
    positions = np.arange(len(data_by_method))
    parts = ax.violinplot(
        [v for _, v in data_by_method],
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        vert=not horizontal,
    )
    for pc, (label, vals) in zip(parts["bodies"], data_by_method):
        pc.set_facecolor(METHODS[[m[1] for m in METHODS].index(label)][2] if label in [m[1] for m in METHODS] else "#888888")
        pc.set_alpha(0.65)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.6)
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels([lbl for lbl, _ in data_by_method], fontsize=7)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels([lbl for lbl, _ in data_by_method], fontsize=7)


def make_scatter_ax(ax, data_by_method, horizontal=False):
    """1D jittered scatter, every client as a dot."""
    for i, (label, vals) in enumerate(data_by_method):
        color = METHODS[[m[1] for m in METHODS].index(label)][2]
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, size=len(vals))
        if horizontal:
            ax.scatter(vals, np.full(len(vals), i) + jitter,
                       s=4, alpha=0.55, color=color, edgecolors="none")
            med = statistics.median(vals)
            ax.vlines(med, i - 0.35, i + 0.35, colors="black", linewidths=1.0)
        else:
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=4, alpha=0.55, color=color, edgecolors="none")
            med = statistics.median(vals)
            ax.hlines(med, i - 0.35, i + 0.35, colors="black", linewidths=1.0)
    if horizontal:
        ax.set_yticks(range(len(data_by_method)))
        ax.set_yticklabels([lbl for lbl, _ in data_by_method], fontsize=7)
    else:
        ax.set_xticks(range(len(data_by_method)))
        ax.set_xticklabels([lbl for lbl, _ in data_by_method], fontsize=7)


def build_grid(dataset_order, k_map, chart_type, horizontal=False):
    n_rows = len(dataset_order)
    n_cols = max(len(v) for v in k_map.values())
    if horizontal:
        # wider, shorter panels to save vertical space
        figsize = (3.1 * n_cols, 1.9 * n_rows + 1.1)
        hspace = 0.45
    else:
        figsize = (2.6 * n_cols, 3.4 * n_rows + 1.1)
        hspace = 0.30
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             squeeze=False, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.17,
                        wspace=0.18, hspace=hspace)
    for r, ds in enumerate(dataset_order):
        ks = k_map[ds]
        for c, k in enumerate(ks):
            ax = axes[r][c]
            rows = []
            for folder, label, _ in METHODS:
                vals = load_final_per_client(ds, folder, k)
                if vals is not None:
                    rows.append((label, vals))
            if chart_type == "violin":
                make_violin_ax(ax, rows, horizontal=horizontal)
            else:
                make_scatter_ax(ax, rows, horizontal=horizontal)
            ax.set_title(f"{DATASET_LABELS[ds]}, $|\\mathcal{{S}}|={k}$", fontsize=9, pad=6)
            if horizontal:
                ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
            else:
                ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
            ax.tick_params(labelsize=7)
        # hide unused columns
        for c in range(len(ks), n_cols):
            axes[r][c].axis("off")
    # accuracy axis label (on the axes, so it stays near the plots)
    if horizontal:
        for c in range(n_cols):
            axes[-1][c].set_xlabel("Client accuracy", fontsize=10)
    else:
        for r in range(n_rows):
            axes[r][0].set_ylabel("Client accuracy", fontsize=10)
    fig.suptitle("Per-client accuracy distribution by method", fontsize=12, y=0.97)
    return fig


def build_legend(fig, horizontal=False):
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                      markersize=8, label=label)
               for folder, label, color in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0),
               borderaxespad=0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset_order = ["cifar10", "cifar100"]
    for opt_name, k_map in [("A", OPTION_A), ("B", OPTION_B)]:
        for chart_type in ["violin", "scatter"]:
            for horizontal in [False, True]:
                fig = build_grid(dataset_order, k_map, chart_type,
                                 horizontal=horizontal)
                build_legend(fig)
                suffix = "_horizontal" if horizontal else ""
                out = os.path.join(
                    OUT_DIR, f"client_dist_option{opt_name}_{chart_type}{suffix}.pdf")
                fig.savefig(out, dpi=300, bbox_inches="tight")
                png = out.replace(".pdf", ".png")
                fig.savefig(png, dpi=150, bbox_inches="tight")
                print(f"saved {out}")
                plt.close(fig)


if __name__ == "__main__":
    main()