#!/usr/bin/env python3
"""Parse training logs in results_combined/ and report wall-clock runtimes.

For each `results_combined/{dataset}/{method}/k{k}/` folder, this reads the
saved log file and extracts the wall-clock time (minutes) to complete the
config's federated rounds.

Method log formats (auto-detected by folder name):
  - fedrl_kl_capped, fedavg : main-runner logs
        start: "=== Starting k={k} | dataset=..."   end: "Training complete!"
  - qfedavg                 : q-FedAvg logs
        start: "=== k={k} | dataset=..."             end: "Round 200/200"
  - favor                   : FAVOR job logs (wall-clock from timestamps)
        boundary: timestamped "Saved -> ...100_clients_{k}_per_round_..."

Usage:
    python3 parse_runtimes.py [--root DIR] [--csv FILE]

Dependencies: Python standard library only.
"""

import os
import re
import glob
import csv
import argparse
from datetime import datetime

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
TS_FMT = "%Y-%m-%d %H:%M:%S,%f"

# method folder name -> display label
METHOD_LABELS = {
    "fedrl_kl_capped": "FedRL",
    "favor": "FAVOR",
    "fedavg": "FedAvg",
    "qfedavg": "q-FedAvg",
}
METHOD_ORDER = ["fedrl_kl_capped", "favor", "fedavg", "qfedavg"]

DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "mnist": "MNIST"}
DATASET_ORDER = ["cifar10", "cifar100", "mnist"]


def parse_ts(s):
    return datetime.strptime(s, TS_FMT)


def pair_times(text, start_re, end_re):
    """Pair timestamped start/end markers into {k: minutes}."""
    out = {}
    cur_k = cur_t = None
    for line in text.splitlines():
        m = TS_RE.match(line.strip())
        if not m:
            continue
        t = parse_ts(m.group(1))
        sm = start_re.search(line)
        if sm:
            cur_k = int(sm.group(1))
            cur_t = t
            continue
        if cur_k is not None and end_re.search(line):
            out[cur_k] = (t - cur_t).total_seconds() / 60.0
            cur_k = cur_t = None
    return out


def main_runner_times(text):
    return pair_times(
        text,
        re.compile(r"=== Starting k=(\d+)"),
        re.compile(r"Training complete!"),
    )


def qfedavg_times(text):
    return pair_times(
        text,
        re.compile(r"=== k=(\d+)\s"),
        re.compile(r"Round 200/200"),
    )


def favor_times(text):
    """FAVOR wall-clock: timestamp of job start -> each k's 'Saved ->' timestamp.

    Each k's warm-up + PCA fit occurs after the previous k's save, so this
    per-k wall-clock naturally includes the setup cost.
    """
    ts_first = None
    saved = []  # (timestamp, k) in order
    saved_re = re.compile(r"Saved .*100_clients_(\d+)_per_round_")
    for line in text.splitlines():
        m = TS_RE.match(line.strip())
        if not m:
            continue
        t = parse_ts(m.group(1))
        if ts_first is None:
            ts_first = t
        sm = saved_re.search(line)
        if sm:
            saved.append((t, int(sm.group(1))))
    out = {}
    prev = ts_first
    for t, k in saved:
        if prev is not None:
            out[k] = (t - prev).total_seconds() / 60.0
        prev = t
    return out


PARSERS = {
    "fedrl_kl_capped": main_runner_times,
    "fedavg": main_runner_times,
    "qfedavg": qfedavg_times,
    "favor": favor_times,
}


def find_log_file(k_dir):
    """Return the first logs_* file in a k folder (None if missing)."""
    hits = sorted(glob.glob(os.path.join(k_dir, "logs_*")))
    return hits[0] if hits else None


def collect(root):
    """Return list of dicts: dataset, method, k, minutes (or None)."""
    rows = []
    for ds in DATASET_ORDER:
        for method in METHOD_ORDER:
            m_dir = os.path.join(root, ds, method)
            if not os.path.isdir(m_dir):
                continue
            for k in [5, 10, 20, 30]:
                k_dir = os.path.join(m_dir, f"k{k}")
                if not os.path.isdir(k_dir):
                    continue
                logf = find_log_file(k_dir)
                if logf is None:
                    rows.append(dict(dataset=ds, method=method, k=k, minutes=None))
                    continue
                with open(logf, "r", errors="replace") as fh:
                    text = fh.read()
                times = PARSERS[method](text)
                rows.append(dict(dataset=ds, method=method, k=k,
                                 minutes=times.get(k)))
    return rows


def report(rows, out=None):
    lines = []
    lines.append("Runtime report - wall-clock minutes to complete 200 rounds")
    lines.append("(single runs; jobs were not hardware-isolated, so absolute")
    lines.append(" times may vary with cluster load)")
    lines.append("")
    lines.append(f"{'Method':<9}{'Dataset':<11}{'k':>4}{'Runtime (min)':>15}")
    lines.append("-" * 39)
    for r in rows:
        val = "INCOMPLETE" if r["minutes"] is None else f"{r['minutes']:.1f}"
        lines.append(
            f"{METHOD_LABELS[r['method']]:<9}{DATASET_LABELS[r['dataset']]:<11}"
            f"{r['k']:>4}{val:>15}"
        )

    # summary: mean over k per (method, dataset)
    lines.append("")
    lines.append("Summary - mean over k=5,10,20,30 (min)")
    hdr = f"{'Method':<9}" + "".join(f"{DATASET_LABELS[d]:>11}" for d in DATASET_ORDER)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for method in METHOD_ORDER:
        cells = []
        for ds in DATASET_ORDER:
            vals = [r["minutes"] for r in rows
                    if r["method"] == method and r["dataset"] == ds
                    and r["minutes"] is not None]
            cells.append(f"{sum(vals)/len(vals):>11.1f}" if vals else f"{'--':>11}")
        lines.append(f"{METHOD_LABELS[method]:<9}" + "".join(cells))

    out_text = "\n".join(lines)
    print(out_text)

    if out:
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "dataset", "k", "runtime_min"])
            for r in rows:
                w.writerow([METHOD_LABELS[r["method"]], DATASET_LABELS[r["dataset"]],
                            r["k"], "" if r["minutes"] is None else f"{r['minutes']:.1f}"])
        print(f"\nCSV written to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="results_combined directory (default: script's own dir)")
    ap.add_argument("--csv", default=None, help="also write a CSV to this path")
    args = ap.parse_args()
    report(collect(args.root), out=args.csv)


if __name__ == "__main__":
    main()
