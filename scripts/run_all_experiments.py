"""
Master Experiment Runner: Executes Experiments 1, 2, 3, and 4 and persists all raw
predictions, metrics, and plot-ready CSV files into results/.
"""
import os, sys, time
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from src.data_generator import generate_retail_network_data
from experiments.exp1_linear_feature_ablation import run_exp1
from experiments.exp2_model_family_benchmark import run_exp2
from experiments.exp3_store_regime_breakdown import run_exp3
from experiments.exp4_public_real_datasets import run_exp4


def export_static_plot_data(repo_root: str):
    """Export all underlying data for Figures 1, 2, 3, and 6 so plotting is 100% decoupled."""
    data = generate_retail_network_data(seed=42)
    plot_dir = os.path.join(repo_root, "results/plot_data")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Fig 1 Topology Nodes & Spatial Matrix
    nodes_rows = []
    for i in range(data["N_stores"]):
        nodes_rows.append({
            "node_id": data["store_ids"][i],
            "node_type": "Store",
            "cluster": data["cluster_names"][i],
            "warehouse": f"W{data['w_assign'][i]+1}",
            "role": data["store_roles"][i],
            "x_km": data["store_coords"][i, 0],
            "y_km": data["store_coords"][i, 1],
        })
    for w in range(data["N_warehouses"]):
        nodes_rows.append({
            "node_id": f"W{w+1}",
            "node_type": "Warehouse",
            "cluster": "Supply_Center",
            "warehouse": f"W{w+1}",
            "role": "Supply_Center",
            "x_km": data["warehouse_coords"][w, 0],
            "y_km": data["warehouse_coords"][w, 1],
        })
    pd.DataFrame(nodes_rows).to_csv(os.path.join(plot_dir, "plot_fig1_topology_nodes.csv"), index=False)

    df_pspa = pd.DataFrame(data["P_spatial"], index=data["store_ids"], columns=data["store_ids"])
    df_pspa.to_csv(os.path.join(plot_dir, "plot_fig1_spatial_matrix.csv"))

    # 2. Fig 2 (184-day 2025 H2 Panorama) & Fig 3 (92-day Q4 S1 & S6 Decomposition)
    full_csv = os.path.join(repo_root, "data/retail_network_2023_2025.csv")
    if os.path.exists(full_csv):
        df_full = pd.read_csv(full_csv)
        df_h2 = df_full[df_full["ds"] >= "2025-07-01"].copy()
        df_h2.to_csv(os.path.join(plot_dir, "plot_fig2_panorama_184d.csv"), index=False)

        df_q4 = df_full[(df_full["ds"] >= "2025-10-01") & (df_full["store_id"].isin(["S1", "S6"]))].copy()
        df_q4.to_csv(os.path.join(plot_dir, "plot_fig3_decomposition_s1_s6.csv"), index=False)

    print("[Plot Data] Exported static plot-ready CSVs to results/plot_data/.")


import argparse
from pathlib import Path
from scripts.plot_all_figures import generate_all_figures, FIGURES_DIR

def main():
    parser = argparse.ArgumentParser(description="Run complete experiment suite.")
    parser.add_argument("--plot", action="store_true", help="Generate all publication-grade figures after experiments.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("=" * 72)
    print("Running Complete Scientific Experiment Suite (Exp 1 -> Exp 4)")
    print("=" * 72)
    export_static_plot_data(repo_root)
    r1 = run_exp1(repo_root)
    r2 = run_exp2(repo_root)
    r3 = run_exp3(repo_root)
    r4 = run_exp4(repo_root)
    el = time.perf_counter() - t0
    print("=" * 72)
    print(f"All Experiments Completed and Persisted to results/ in {el:.2f}s!")
    print("=" * 72)

    if args.plot:
        generate_all_figures(FIGURES_DIR, ["zh", "en"])


if __name__ == "__main__":
    main()
