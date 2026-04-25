#!/usr/bin/env python3
"""
Pearson Correlation Analysis: Participation Frequency vs Client Accuracy Contribution
Compares ablated vs FedRL strategies across different client selection amounts.
"""

import os
import json
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

# ===================== CONFIG =====================
BASE_DIR = "."
ABLATED_DIR = os.path.join(BASE_DIR, "results_for_runs_cifar_fairness_abalated")
FEDRL_DIR = os.path.join(BASE_DIR, "results_for_runs_cifar_fairness_epsilon_0.5")

CLIENT_COUNTS = [5, 10, 20, 30]

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

def extract_correlation_data(participation_freq, processed_acc):
    """
    Extract participation and accuracy arrays for correlation calculation.
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

def calculate_pearson(x, y):
    """Calculate Pearson correlation coefficient and p-value"""
    if x is None or y is None or len(x) < 2:
        return None, None
    
    try:
        corr, p_value = pearsonr(x, y)
        return corr, p_value
    except Exception as e:
        print(f"[WARN] Failed to calculate Pearson correlation: {e}")
        return None, None

# ===================== ANALYSIS =====================
def analyze_correlations():
    """Calculate and compare Pearson correlations for all client counts"""
    
    results = []
    
    print("\n" + "="*80)
    print("PEARSON CORRELATION ANALYSIS: Participation Frequency vs Client Accuracy")
    print("="*80)
    
    for m in CLIENT_COUNTS:
        sub = subdir_name(m)
        ablated_path = os.path.join(ABLATED_DIR, sub, "run_results.json")
        fedrl_path = os.path.join(FEDRL_DIR, sub, "run_results.json")
        
        if not os.path.isfile(ablated_path) or not os.path.isfile(fedrl_path):
            print(f"[WARN] Missing data files for m={m}")
            continue
        
        # Load ablated data
        ablated_part, ablated_acc = load_client_data(ablated_path)
        x_ablated, y_ablated = extract_correlation_data(ablated_part, ablated_acc)
        corr_ablated, p_ablated = calculate_pearson(x_ablated, y_ablated)
        
        # Load FedRL data
        fedrl_part, fedrl_acc = load_client_data(fedrl_path)
        x_fedrl, y_fedrl = extract_correlation_data(fedrl_part, fedrl_acc)
        corr_fedrl, p_fedrl = calculate_pearson(x_fedrl, y_fedrl)
        
        # Store results
        results.append({
            "Client Count": m,
            "Ablated Corr": corr_ablated,
            "Ablated p-value": p_ablated,
            "FedRL Corr": corr_fedrl,
            "FedRL p-value": p_fedrl,
            "Correlation Diff": corr_fedrl - corr_ablated if (corr_ablated is not None and corr_fedrl is not None) else None,
        })
        
        # Print results for this client count
        print(f"\n|𝒮_t| = {m} clients/round:")
        print("-" * 80)
        
        if corr_ablated is not None:
            sig_ablated = "***" if p_ablated < 0.001 else "**" if p_ablated < 0.01 else "*" if p_ablated < 0.05 else "ns"
            print(f"  Ablated:  r = {corr_ablated:7.4f}, p-value = {p_ablated:.6f} {sig_ablated}")
        else:
            print(f"  Ablated:  Failed to calculate")
        
        if corr_fedrl is not None:
            sig_fedrl = "***" if p_fedrl < 0.001 else "**" if p_fedrl < 0.01 else "*" if p_fedrl < 0.05 else "ns"
            print(f"  FedRL:    r = {corr_fedrl:7.4f}, p-value = {p_fedrl:.6f} {sig_fedrl}")
        else:
            print(f"  FedRL:    Failed to calculate")
        
        if corr_ablated is not None and corr_fedrl is not None:
            diff = corr_fedrl - corr_ablated
            print(f"  Δ (FedRL - Ablated): {diff:7.4f}")
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Export to CSV
    csv_path = "pearson_correlation_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to {csv_path}")
    
    return df

if __name__ == "__main__":
    analyze_correlations()
