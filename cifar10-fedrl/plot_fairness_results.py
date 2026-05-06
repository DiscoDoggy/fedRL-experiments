#!/usr/bin/env python3
"""
Fairness Analysis Plot — Per-Client Accuracy Distributions

For each client subset size k in [5, 10, 20, 30], produces:
  1. accuracy_vs_rounds.pdf     — global accuracy curves per method
  2. final_round_violin.pdf     — violin of per-client accs at best round
  3. variance_vs_rounds.pdf     — per-client accuracy variance over rounds
  4. summary_stats.csv/json     — numerical summary table

Output layout:
  fairness_results_multiple_client_sizes/plots/k_<N>/
"""

import csv
import os
import json
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE  = os.path.join(SCRIPT_DIR, "fairness_results_multiple_client_sizes")
PLOTS_BASE    = os.path.join(RESULTS_BASE, "plots")
K_VALUES      = [5, 10, 20, 30]

METHODS = ["fedrl", "fedavg", "ablated", "flash_rl"]
METHOD_LABELS = {"fedrl": "FedRL", "fedavg": "FedAvg", "ablated": "Ablated", "flash_rl": "FLASH-RL"}
COLORS = {"fedrl": "#1f77b4", "fedavg": "#d62728", "ablated": "#ff7f0e", "flash_rl": "#2ca02c"}

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
})


def load(method, k):
    path = os.path.join(RESULTS_BASE, method, f"k_{k}", "fairness_results.json")
    if not os.path.isfile(path):
        print(f"  [SKIP] {method} k={k}: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def per_client_matrix(data):
    """Returns ndarray shape (num_rounds, num_clients)."""
    rounds = data["per_client_accuracies"]
    # all client ids present across all rounds
    all_ids = sorted({int(k) for rd in rounds for k in rd})
    mat = np.array([
        [rd.get(str(cid), np.nan) for cid in all_ids]
        for rd in rounds
    ])
    return mat


# ---- 1. Global accuracy vs rounds ----
def plot_global_accuracy(datasets, out_dir, k):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, d in datasets.items():
        accs = d["global_accuracies"]
        rounds = range(1, len(accs) + 1)
        ax.plot(rounds, accs, color=COLORS[method], linewidth=2,
                label=METHOD_LABELS[method])
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Accuracy")
    ax.set_title(f"Global Accuracy vs Rounds — CIFAR-100 ($|S_t|={k}$)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(out_dir, "accuracy_vs_rounds.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---- 2. Violin plot of per-client accuracy at best round ----
def plot_final_violin(datasets, out_dir, k):
    fig, ax = plt.subplots(figsize=(7, 5))
    all_accs = []
    positions = []
    labels = []
    for i, (method, d) in enumerate(datasets.items()):
        mat = per_client_matrix(d)
        best_round_idx = int(np.argmax(d["global_accuracies"]))
        final_accs = mat[best_round_idx]
        final_accs = final_accs[~np.isnan(final_accs)]
        all_accs.append(final_accs)
        positions.append(i + 1)
        labels.append(METHOD_LABELS[method])

    parts = ax.violinplot(all_accs, positions=positions, showmedians=True,
                          showextrema=True)
    for i, (pc, method) in enumerate(zip(parts['bodies'], datasets.keys())):
        pc.set_facecolor(COLORS[method])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cbars'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-Client Local Accuracy (Best Round)")
    ax.set_title(f"Per-Client Accuracy Distribution ($|S_t|={k}$)")
    ax.grid(True, axis='y', linestyle="--", alpha=0.4)

    for i, accs in enumerate(all_accs):
        ax.text(positions[i], ax.get_ylim()[0] + 0.01,
                f"σ²={np.var(accs):.4f}\nmin={np.min(accs):.3f}",
                ha='center', fontsize=10, color='#333333')

    fig.tight_layout()
    out = os.path.join(out_dir, "final_round_violin.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---- 3. Accuracy variance over rounds ----
def plot_variance_over_rounds(datasets, out_dir, k):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, d in datasets.items():
        mat = per_client_matrix(d)
        variances = np.nanvar(mat, axis=1)
        rounds = range(1, len(variances) + 1)
        ax.plot(rounds, variances, color=COLORS[method], linewidth=2,
                label=METHOD_LABELS[method])
    ax.set_xlabel("Round")
    ax.set_ylabel("Variance of Per-Client Accuracy")
    ax.set_title(f"Per-Client Accuracy Variance Over Rounds ($|S_t|={k}$)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(out_dir, "variance_vs_rounds.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---- 4. Summary stats table + CSV save ----
def print_summary(datasets, out_dir, k):
    headers = ["Method", "Global Best", "Mean Local",
               "Var All", "Min Local", "Max Local", "Worst 10% Avg", "Best 10% Avg"]
    rows = []

    print(f"\n  k={k}")
    print(f"  {'Method':<12} {'GBest':>8} {'MeanLoc':>9} {'VarAll':>10} "
          f"{'Min':>8} {'Max':>8} {'W10%Avg':>9} {'B10%Avg':>9}")
    print("  " + "-" * 80)

    for method, d in datasets.items():
        mat = per_client_matrix(d)
        best_round_idx = int(np.argmax(d["global_accuracies"]))
        final = mat[best_round_idx]
        final = final[~np.isnan(final)]
        sorted_accs = np.sort(final)
        n10 = max(1, int(len(sorted_accs) * 0.1))

        global_best   = max(d["global_accuracies"])
        mean_local    = np.mean(final)
        var_all       = np.var(final)
        var_worst10   = np.var(sorted_accs[:n10])
        var_best10    = np.var(sorted_accs[-n10:])
        min_local     = np.min(final)
        max_local     = np.max(final)
        worst10_avg   = np.mean(sorted_accs[:n10])
        best10_avg    = np.mean(sorted_accs[-n10:])

        print(f"  {METHOD_LABELS[method]:<12} {global_best:>8.4f} {mean_local:>9.4f} "
              f"{var_all:>10.5f} "
              f"{min_local:>8.4f} {max_local:>8.4f} {worst10_avg:>9.4f} {best10_avg:>9.4f}")

        rows.append({
            "Method":        METHOD_LABELS[method],
            "Global Best":   round(global_best,  4),
            "Mean Local":    round(mean_local,   4),
            "Var All":       round(var_all,      6),
            "Min Local":     round(min_local,    4),
            "Max Local":     round(max_local,    4),
            "Worst 10% Avg": round(worst10_avg,  4),
            "Best 10% Avg":  round(best10_avg,   4),
        })

    csv_path = os.path.join(out_dir, "summary_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")

    json_path = os.path.join(out_dir, "summary_stats.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved: {json_path}")
    return rows


# ---- 5. Best-client accuracy distribution across k values ----
def plot_best_client_distribution():
    """
    For each k value, collect the best client accuracy at every round
    (max over all 100 clients). Plots a violin per k, one subplot per method.
    Saved to PLOTS_BASE/best_client_distribution.pdf/png
    """
    # Gather data: method -> k -> array of shape (num_rounds,)
    all_data = {}
    for method in METHODS:
        method_data = {}
        for k in K_VALUES:
            d = load(method, k)
            if d is None:
                continue
            mat = per_client_matrix(d)
            # max accuracy across clients at each round
            best_per_round = np.nanmax(mat, axis=1)
            method_data[k] = best_per_round
        if method_data:
            all_data[method] = method_data

    if not all_data:
        print("  No data available for best-client distribution plot.")
        return

    n_methods = len(all_data)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, (method, method_data) in zip(axes, all_data.items()):
        ks = sorted(method_data.keys())
        accs_list = [method_data[k] for k in ks]
        positions = list(range(1, len(ks) + 1))

        parts = ax.violinplot(accs_list, positions=positions,
                              showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS[method])
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cbars'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cmaxes'].set_color('black')

        ax.set_xticks(positions)
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("Clients per Round ($|S_t|$)")
        ax.grid(True, axis='y', linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Best Client Local Accuracy (per Round)")
    fig.suptitle("Distribution of Best Client Accuracy per Round", fontsize=15)
    fig.tight_layout()

    os.makedirs(PLOTS_BASE, exist_ok=True)
    out = os.path.join(PLOTS_BASE, "best_client_distribution.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---- 6. Per-client best accuracy strip plot (one dot per client) ----
def plot_per_client_best_strip():
    """
    For each method, plots a horizontal strip chart where each dot is one client's
    best accuracy (max over all rounds). One row per k value.
    Saved to PLOTS_BASE/per_client_best_strip_<method>.pdf/png
    """
    rng = np.random.default_rng(0)

    for method in METHODS:
        rows_data = {}
        for k in K_VALUES:
            d = load(method, k)
            if d is None:
                continue
            mat = per_client_matrix(d)
            # best accuracy per client across all rounds
            best_per_client = np.nanmax(mat, axis=0)
            rows_data[k] = best_per_client

        if not rows_data:
            continue

        ks = sorted(rows_data.keys())
        fig, ax = plt.subplots(figsize=(8, 3 + len(ks) * 0.8))

        for i, k in enumerate(ks):
            accs = rows_data[k]
            # jitter on y-axis for readability
            y = np.full(len(accs), i) + rng.uniform(-0.15, 0.15, size=len(accs))
            ax.scatter(accs, y, alpha=0.55, s=18, color=COLORS[method], linewidths=0)
            # overlay median line
            ax.plot([np.median(accs), np.median(accs)],
                    [i - 0.3, i + 0.3], color='black', linewidth=1.5, zorder=3)

        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([str(k) for k in ks])
        ax.set_ylabel("Number of clients selected in each round", rotation=90, labelpad=10)
        ax.set_xlabel("Best Client Accuracy (max over all rounds)")
        ax.set_title(f"Per-Client Best Accuracy — {METHOD_LABELS[method]}")
        ax.grid(True, axis='x', linestyle="--", alpha=0.4)
        fig.tight_layout()

        os.makedirs(PLOTS_BASE, exist_ok=True)
        out = os.path.join(PLOTS_BASE, f"per_client_best_strip_{method}.pdf")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ---- 7b. Combined 2×2 strip chart (k=5 and k=10 only) ----
def plot_combined_strip_2x2():
    """
    2×2 subplot grid — one panel per method (fedrl, fedavg, ablated, flash_rl).
    Within each panel: two rows, k=5 (top) and k=10 (bottom).
    Each dot = one client's best accuracy across all rounds.
    Saved to PLOTS_BASE/combined_strip_2x2.pdf/png
    """
    K_SUBSET = [5, 10]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    axes_flat = axes.flatten()

    for ax, method in zip(axes_flat, METHODS):
        color = COLORS[method]
        plotted_ks = []
        for k in K_SUBSET:
            d = load(method, k)
            if d is None:
                continue
            mat = per_client_matrix(d)
            best_per_client = np.nanmax(mat, axis=0)
            i = K_SUBSET.index(k)
            y = np.full(len(best_per_client), i) + rng.uniform(-0.15, 0.15, size=len(best_per_client))
            ax.scatter(best_per_client, y, alpha=0.55, s=14, color=color, linewidths=0)
            ax.plot([np.median(best_per_client)] * 2, [i - 0.3, i + 0.3],
                    color='black', linewidth=1.8, zorder=3)
            plotted_ks.append(k)

        ax.set_yticks(range(len(K_SUBSET)))
        ax.set_yticklabels([f"$|S_t|={k}$" for k in K_SUBSET])
        ax.set_xlabel("Best client accuracy (max over rounds)")
        ax.set_title(METHOD_LABELS[method], fontweight="bold")
        ax.grid(True, axis='x', linestyle="--", alpha=0.4)
        ax.set_xlim(0.2, 0.75)

    fig.suptitle("Per-Client Best Accuracy Distribution (CIFAR-100, α=0.5, 100 clients)",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    os.makedirs(PLOTS_BASE, exist_ok=True)
    out = os.path.join(PLOTS_BASE, "combined_strip_2x2.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---- 7c. Pairwise 2×1 strip charts ----
def plot_pairwise_strip_2x1(method_a, method_b):
    """
    Side-by-side (1 row, 2 cols) strip chart comparing two methods.
    Each panel: two rows for k=5 (top) and k=10 (bottom).
    Saved to PLOTS_BASE/strip_<method_a>_vs_<method_b>.pdf/png
    """
    K_SUBSET = [5, 10]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, method in zip(axes, [method_a, method_b]):
        color = COLORS[method]
        for i, k in enumerate(K_SUBSET):
            d = load(method, k)
            if d is None:
                continue
            mat = per_client_matrix(d)
            best_per_client = np.nanmax(mat, axis=0)
            y = np.full(len(best_per_client), i) + rng.uniform(-0.15, 0.15, size=len(best_per_client))
            ax.scatter(best_per_client, y, alpha=0.55, s=16, color=color, linewidths=0)
            ax.plot([np.median(best_per_client)] * 2, [i - 0.3, i + 0.3],
                    color='black', linewidth=1.8, zorder=3)

        ax.set_yticks(range(len(K_SUBSET)))
        ax.set_yticklabels([f"$|S_t|={k}$" for k in K_SUBSET])
        ax.set_xlabel("Best client accuracy (max over rounds)")
        ax.set_title(METHOD_LABELS[method], fontweight="bold")
        ax.grid(True, axis='x', linestyle="--", alpha=0.4)
        ax.set_xlim(0.2, 0.75)

    fig.suptitle(
        f"Per-Client Best Accuracy: {METHOD_LABELS[method_a]} vs {METHOD_LABELS[method_b]}\n"
        f"(CIFAR-100, α=0.5, 100 clients)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    os.makedirs(PLOTS_BASE, exist_ok=True)
    out = os.path.join(PLOTS_BASE, f"strip_{method_a}_vs_{method_b}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---- 7. Cross-k fairness table ----
def plot_variance_table():
    """
    Produces a CSV + printed table with columns:
      Method | k | Var All | Worst 10% Avg | Best 10% Avg
    across all method × k combinations.
    Saved to RESULTS_BASE/plots/variance_table.csv
    """
    headers = ["Method", "k", "Var All", "Worst 10% Avg", "Best 10% Avg"]
    rows = []

    print(f"\n  {'Method':<12} {'k':>4} {'Var All':>12} {'Worst 10% Avg':>14} {'Best 10% Avg':>13}")
    print("  " + "-" * 58)

    for method in METHODS:
        for k in K_VALUES:
            d = load(method, k)
            if d is None:
                continue
            mat = per_client_matrix(d)
            best_round_idx = int(np.argmax(d["global_accuracies"]))
            final = mat[best_round_idx]
            final = final[~np.isnan(final)]
            sorted_accs = np.sort(final)
            n10 = max(1, int(len(sorted_accs) * 0.1))

            var_all     = np.var(final)
            worst10_avg = np.mean(sorted_accs[:n10])
            best10_avg  = np.mean(sorted_accs[-n10:])

            print(f"  {METHOD_LABELS[method]:<12} {k:>4} {var_all:>12.6f} "
                  f"{worst10_avg:>14.4f} {best10_avg:>13.4f}")
            rows.append({
                "Method":        METHOD_LABELS[method],
                "k":             k,
                "Var All":       round(var_all,      6),
                "Worst 10% Avg": round(worst10_avg,  4),
                "Best 10% Avg":  round(best10_avg,   4),
            })

    cross_dir = os.path.join(RESULTS_BASE, "plots")
    os.makedirs(cross_dir, exist_ok=True)
    csv_path = os.path.join(cross_dir, "variance_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    json_path = csv_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved: {json_path}")


def main():
    for k in K_VALUES:
        print(f"\n{'='*60}\n  Plotting k={k}\n{'='*60}")
        out_dir = os.path.join(PLOTS_BASE, f"k_{k}")
        os.makedirs(out_dir, exist_ok=True)

        datasets = {}
        for method in METHODS:
            d = load(method, k)
            if d is not None:
                datasets[method] = d

        if not datasets:
            print(f"  No results found for k={k}, skipping.")
            continue

        plot_global_accuracy(datasets, out_dir, k)
        plot_final_violin(datasets, out_dir, k)
        plot_variance_over_rounds(datasets, out_dir, k)
        print_summary(datasets, out_dir, k)

    print(f"\nGenerating best-client distribution plot across all k values...")
    plot_best_client_distribution()

    print("\nGenerating per-client best accuracy strip plots...")
    plot_per_client_best_strip()

    print("\nGenerating cross-k variance table...")
    plot_variance_table()

    print("\nGenerating 2x2 combined strip plot (k=5,10)...")
    plot_combined_strip_2x2()

    print("\nGenerating 2x1 pairwise strip plots (k=5,10)...")
    for pair in [("fedrl", "fedavg"), ("fedrl", "ablated"), ("fedrl", "flash_rl")]:
        plot_pairwise_strip_2x1(*pair)

    print("\nDone.")


if __name__ == "__main__":
    main()
