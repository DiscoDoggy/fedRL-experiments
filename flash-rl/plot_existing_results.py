#!/usr/bin/env python3
"""Generate plots for existing FLASH-RL results without re-running experiments."""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_rl_main import save_plots, jain_fairness_index


def generate_plots(result_dir, dataset_name):
    for item in sorted(os.listdir(result_dir)):
        if not item.startswith("100_clients_"):
            continue
        sub = os.path.join(result_dir, item)
        jp = os.path.join(sub, "run_results.json")
        if not os.path.isfile(jp):
            print(f"  Skipping {item}: no run_results.json")
            continue

        d = json.load(open(jp))
        pc = d.get("per_client_accuracies", [])
        if not pc or len(pc) == 0:
            print(f"  Skipping {item}: no per_client_accuracies")
            continue

        means = [sum(r) / len(r) for r in pc]
        stds = [float(np.std(r)) for r in pc]
        jfis = [jain_fairness_index(r) for r in pc]

        plots_dir = os.path.join(sub, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        num_clients = len(pc[0])
        save_plots(means, stds, jfis, d.get("participation_freq", {}),
                   num_clients, dataset_name, plots_dir,
                   all_per_client_accs=pc)
        print(f"  ✓ {item}: {len(pc)} rounds, {num_clients} clients, plots → {plots_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_existing_results.py <result_dir1> [result_dir2 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if not os.path.isdir(path):
            print(f"Not a directory: {path}")
            continue
        # Derive dataset name from path
        name = "cifar10" if "cifar10" in path.lower() else "cifar100" if "cifar100" in path.lower() else path
        print(f"\n=== {path} ===")
        generate_plots(path, name)
