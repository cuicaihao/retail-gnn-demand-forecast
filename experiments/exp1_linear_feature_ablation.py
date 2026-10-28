"""
Experiment 1: Linear Feature Dimension & Sample Pooling Ablation Ladder (L0 -> L4).
Strictly isolates:
- Uncensoring effect (L0 -> L1)
- Sample pooling effect without graph (L1 -> L2a / L2b)
- 1-Hop linear neighborhood effect (L1 -> L3 / L3p)
- 2-Hop conditional interaction effect (L3 -> L4)
Persists all macro, micro (per-store), and plot-ready CSVs to results/.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_generator import generate_retail_network_data
from src.feature_builders import build_all_feature_sets
from src.models_linear import fit_predict_linear_spec, fit_predict_tobit_censored_mle
from src.evaluation import evaluate_model_macro_and_micro


def run_exp1(repo_root: str) -> dict:
    data = generate_retail_network_data(seed=42)
    feats = build_all_feature_sets(data)

    ladder_specs = [
        ("L0_1A_Local_Naive", "1A. 单店全量基线 (Local Naive OLS)", "Local (Per-Store)", "All Open (10,620)"),
        ("L1_1B_Local_Uncensored", "1B. 单店去截断基线 (Local Uncensored OLS)", "Local (Per-Store)", "Uncensored (8,835)"),
        ("L1_Tobit_Censored_MLE", "1B-Tobit. 单店截断似然 (Right-Censored Tobit MLE)", "Local (Per-Store)", "All Open w/ Censored Likelihood"),
        ("L2a_1Bp_Pooled_Shared", "1B'-Shared. 全网池化共享弹性 (Pooled No-Graph, 16d)", "Pooled (10 Stores)", "Uncensored (8,835)"),
        ("L2b_1Bp_Pooled_BlockDiag", "1B'-Block. 全网池化独立弹性 (Pooled Block-Diag, 70d)", "Pooled (10 Stores)", "Uncensored (8,835)"),
        ("L3_2_Linear_1Hop_Local", "2-Local. 单店一阶邻域线性模型 (1-Hop Flat Local, 9d)", "Local (Per-Store)", "Uncensored (8,835)"),
        ("L3p_2_Linear_1Hop_Pooled", "2-Pooled. 全网池化一阶邻域模型 (1-Hop Flat Pooled, 72d)", "Pooled (10 Stores)", "Uncensored (8,835)"),
        ("L4_3_Analytical_2Hop_Graph", "3 (2+). 两跳条件交互解析图模型 (2-Hop Analytical Graph, 79d)", "Pooled (10 Stores)", "Uncensored (8,835)"),
    ]

    macro_list = []
    micro_list = []
    preds_dict = {}

    for code_id, label, pool_mode, sample_desc in ladder_specs:
        if code_id == "L1_Tobit_Censored_MLE":
            f_spec = feats["L0_1A_Local_Naive"]
            pred_d, pred_s, rt = fit_predict_tobit_censored_mle(f_spec, data)
            active_d = 7
            mat_cols = 7
            free_p = 80
            f_names = "; ".join(f_spec["feature_names"]) + "; log_sigma"
        else:
            f_spec = feats[code_id]
            pred_d, pred_s, rt, _ = fit_predict_linear_spec(code_id, f_spec, data)
            active_d = f_spec["active_dim"]
            mat_cols = f_spec["matrix_cols"]
            free_p = f_spec["free_params"]
            f_names = "; ".join(f_spec["feature_names"][:12]) + (" ..." if len(f_spec["feature_names"]) > 12 else "")

        preds_dict[code_id] = (pred_d, pred_s)
        meta = {
            "display_name": label,
            "pooling_strategy": pool_mode,
            "training_samples": sample_desc,
            "active_dim_per_store": active_d,
            "design_matrix_cols": mat_cols,
            "total_free_params": free_p,
            "runtime_sec": round(rt, 5),
            "feature_list_summary": f_names,
        }
        m_macro, m_micro = evaluate_model_macro_and_micro(code_id, pred_d, data, pred_s, meta=meta)
        macro_list.append(m_macro)
        micro_list.extend(m_micro)

    df_macro = pd.DataFrame(macro_list)
    df_micro = pd.DataFrame(micro_list)

    df_macro.to_csv(os.path.join(repo_root, "results/metrics/exp1_feature_ablation_metrics.csv"), index=False)
    df_micro.to_csv(os.path.join(repo_root, "results/metrics/exp1_feature_ablation_by_store.csv"), index=False)
    df_macro.to_csv(os.path.join(repo_root, "results/plot_data/plot_fig8_feature_dim_vs_wape.csv"), index=False)

    print("[Exp 1] Completed Linear Feature & Pooling Ablation Ladder (L0 -> L4).")
    return {"macro": df_macro, "micro": df_micro, "preds": preds_dict}


if __name__ == "__main__":
    run_exp1(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
