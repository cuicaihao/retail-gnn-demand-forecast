"""
Experiment 2: Full Network Cross-Model Family Benchmark (Linear vs. Tree/GBDT vs. Neural GNN)
+ Negative Control (E5: zero coupling) + Coupling Phase Sweep (E6).
Persists all predictions, multi-seed mean/std metrics, feature importances, and plot data.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_generator import generate_retail_network_data
from src.feature_builders import build_all_feature_sets
from src.models_linear import fit_predict_linear_spec, fit_predict_tobit_censored_mle
from src.models_tree import fit_predict_gbdt_multi_seed
from src.models_gnn import fit_predict_neural_gnn_multi_seed
from src.models_prophet import fit_predict_prophet_spatial_ladder
from src.evaluation import evaluate_model_macro_and_micro, get_test_regime_masks, compute_metrics


def run_exp2(repo_root: str) -> dict:
    seeds = [42, 101, 202, 303, 404]
    data = generate_retail_network_data(seed=42)
    feats = build_all_feature_sets(data)
    masks = get_test_regime_masks(data)

    # 1. Fit Linear / Statistical Baseline Models
    p_1a, s_1a, rt_1a, _ = fit_predict_linear_spec("L0_1A", feats["L0_1A_Local_Naive"], data)
    p_1b, s_1b, rt_1b, _ = fit_predict_linear_spec("L1_1B", feats["L1_1B_Local_Uncensored"], data)
    p_tobit, s_tobit, rt_tobit = fit_predict_tobit_censored_mle(feats["L0_1A_Local_Naive"], data)
    p_l2a, s_l2a, rt_l2a, _ = fit_predict_linear_spec("L2a", feats["L2a_1Bp_Pooled_Shared"], data)
    p_l3, s_l3, rt_l3, _ = fit_predict_linear_spec("L3", feats["L3_2_Linear_1Hop_Local"], data)
    p_l4, s_l4, rt_l4, _ = fit_predict_linear_spec("L4", feats["L4_3_Analytical_2Hop_Graph"], data)

    # 2. Fit Tree / GBDT Models (LightGBM & XGBoost across 5 seeds: Base, 1Hop, 2Hop)
    gbdt_11d = fit_predict_gbdt_multi_seed(feats["Tree_GBDT_Base_11d"], data, seeds=seeds, engine="lgbm")
    xgb_11d = fit_predict_gbdt_multi_seed(feats["Tree_GBDT_Base_11d"], data, seeds=seeds, engine="xgboost")
    gbdt_18d = fit_predict_gbdt_multi_seed(feats["Tree_GBDT_1Hop_18d"], data, seeds=seeds, engine="lgbm")
    xgb_18d = fit_predict_gbdt_multi_seed(feats["Tree_GBDT_1Hop_18d"], data, seeds=seeds, engine="xgboost")
    gbdt_27d = fit_predict_gbdt_multi_seed(feats["Tree_GBDT_2Hop_27d"], data, seeds=seeds, engine="lgbm")
    xgb_27d = fit_predict_gbdt_multi_seed(feats["Tree_GBDT_2Hop_27d"], data, seeds=seeds, engine="xgboost")

    # 3. Fit Prophet Additive Time-Series Ladder (Base, 1Hop, 2Hop)
    prophet_results = fit_predict_prophet_spatial_ladder(data)
    prop_base = prophet_results["Prophet_Base"]
    prop_1h = prophet_results["Prophet_1Hop"]
    prop_2h = prophet_results["Prophet_2Hop"]

    # 4. Fit End-to-End Neural 2-Layer Hetero-GNN across 5 seeds
    neural_gnn = fit_predict_neural_gnn_multi_seed(data, seeds=seeds, hidden_dim=16, epochs=350)

    # Save Feature Importances for Tree Models (18d and 27d)
    df_imp_18 = pd.DataFrame({
        "feature_name": feats["Tree_GBDT_1Hop_18d"]["feature_names"],
        "importance_gain_share": np.round(gbdt_18d["feature_importances"] * 100.0, 2),
    }).sort_values("importance_gain_share", ascending=False)
    df_imp_18.to_csv(os.path.join(repo_root, "results/metrics/exp2_gbdt_18d_feature_importance.csv"), index=False)

    df_imp_xgb = pd.DataFrame({
        "feature_name": feats["Tree_GBDT_1Hop_18d"]["feature_names"],
        "importance_gain_share": np.round(xgb_18d["feature_importances"] * 100.0, 2),
    }).sort_values("importance_gain_share", ascending=False)
    df_imp_xgb.to_csv(os.path.join(repo_root, "results/metrics/exp2_xgboost_18d_feature_importance.csv"), index=False)

    df_imp_27 = pd.DataFrame({
        "feature_name": feats["Tree_GBDT_2Hop_27d"]["feature_names"],
        "importance_gain_share": np.round(gbdt_27d["feature_importances"] * 100.0, 2),
    }).sort_values("importance_gain_share", ascending=False)
    df_imp_27.to_csv(os.path.join(repo_root, "results/metrics/exp2_gbdt_27d_feature_importance.csv"), index=False)

    df_imp_xgb_27 = pd.DataFrame({
        "feature_name": feats["Tree_GBDT_2Hop_27d"]["feature_names"],
        "importance_gain_share": np.round(xgb_27d["feature_importances"] * 100.0, 2),
    }).sort_values("importance_gain_share", ascending=False)
    df_imp_xgb_27.to_csv(os.path.join(repo_root, "results/metrics/exp2_xgboost_27d_feature_importance.csv"), index=False)

    # Collect all models for unified macro & micro evaluation using standardized [Model]-[Data/Tier] naming
    model_entries = [
        ("M1A_OLS_Naive_7d", "Linear/Stat", "OLS-Naive (7d 全量含缺货)", p_1a, s_1a, None, 7, 7, 70, rt_1a),
        ("M1B_OLS_Uncens_7d", "Linear/Stat", "OLS-Base (7d 单店去截断)", p_1b, s_1b, None, 7, 7, 70, rt_1b),
        ("M1B_Tobit_MLE_7d", "Linear/Stat", "Tobit-MLE (7d 截断似然)", p_tobit, s_tobit, None, 7, 7, 80, rt_tobit),
        ("M1Bp_Pooled_NoGraph_16d", "Linear/Stat", "OLS-Pooled (16d 全网池化无图)", p_l2a, s_l2a, None, 7, 16, 16, rt_l2a),
        ("M2_OLS_1Hop_9d", "Linear/Stat", "OLS-1Hop (9d 一阶邻域线性)", p_l3, s_l3, None, 9, 9, 90, rt_l3),
        ("M_Prophet_Base", "GAM/Prophet", "Prophet-Base (2d 外生 + 原生周/年傅里叶)", prop_base["pred_demand"], prop_base["pred_sales"], None, 2, 2, prop_base["total_params"], prop_base["runtime_sec"]),
        ("M_Prophet_1Hop", "GAM/Prophet", "Prophet-1Hop (5d 外生宽表 + 原生周/年傅里叶)", prop_1h["pred_demand"], prop_1h["pred_sales"], None, 5, 5, prop_1h["total_params"], prop_1h["runtime_sec"]),
        ("M_Prophet_2Hop", "GAM/Prophet", "Prophet-2Hop (7d 二阶拓扑外生 + 原生傅里叶)", prop_2h["pred_demand"], prop_2h["pred_sales"], None, 7, 7, prop_2h["total_params"], prop_2h["runtime_sec"]),
        ("M_GBDT_Base_11d", "Tree/LightGBM", "LightGBM-Base (11d 单店无图基线)", gbdt_11d["mean_pred_demand"], gbdt_11d["mean_pred_sales"], gbdt_11d["seed_preds"], 11, 11, 4960, gbdt_11d["mean_runtime_sec"]),
        ("M_GBDT_1Hop_Raw_18d", "Tree/LightGBM", "LightGBM-1Hop (18d 一阶邻域宽表)", gbdt_18d["mean_pred_demand"], gbdt_18d["mean_pred_sales"], gbdt_18d["seed_preds"], 18, 18, 4960, gbdt_18d["mean_runtime_sec"]),
        ("M_GBDT_2Hop_Graph_27d", "Tree/LightGBM", "LightGBM-2Hop (27d 显式二阶图特征)", gbdt_27d["mean_pred_demand"], gbdt_27d["mean_pred_sales"], gbdt_27d["seed_preds"], 27, 27, 4960, gbdt_27d["mean_runtime_sec"]),
        ("M_XGBoost_Base_11d", "Tree/XGBoost", "XGBoost-Base (11d 经典单店基线)", xgb_11d["mean_pred_demand"], xgb_11d["mean_pred_sales"], xgb_11d["seed_preds"], 11, 11, 4960, xgb_11d["mean_runtime_sec"]),
        ("M_XGBoost_1Hop_Raw_18d", "Tree/XGBoost", "XGBoost-1Hop (18d 经典一阶宽表)", xgb_18d["mean_pred_demand"], xgb_18d["mean_pred_sales"], xgb_18d["seed_preds"], 18, 18, 4960, xgb_18d["mean_runtime_sec"]),
        ("M_XGBoost_2Hop_Graph_27d", "Tree/XGBoost", "XGBoost-2Hop (27d 经典显式二阶图)", xgb_27d["mean_pred_demand"], xgb_27d["mean_pred_sales"], xgb_27d["seed_preds"], 27, 27, 4960, xgb_27d["mean_runtime_sec"]),
        ("M3_Neural_HeteroGNN_2Layer", "Neural GNN", "NeuralGNN-2Hop (两层异构神经图网络)", neural_gnn["mean_pred_demand"], neural_gnn["mean_pred_sales"], neural_gnn["seed_preds"], 10, 24, neural_gnn["total_params"], neural_gnn["mean_runtime_sec"]),
        ("M3_Analytical_2Hop_79d", "Analytical Graph", "Analytical-2Hop (79d 两层解析图模型 · 理论参照上限)", p_l4, s_l4, None, 16, 79, 79, rt_l4),
    ]

    macro_rows = []
    micro_rows = []

    for m_id, fam, label, pd_arr, ps_arr, sd_preds, act_d, mat_c, n_par, rt_s in model_entries:
        meta = {
            "family": fam,
            "display_name": label,
            "active_dim_per_store": act_d,
            "design_matrix_cols": mat_c,
            "total_params": n_par,
            "runtime_sec": round(rt_s, 4),
        }
        # Compute seed std if multi-seed
        if sd_preds is not None:
            for r_name, r_mask in masks.items():
                wapes_sd = [compute_metrics(data["D_sim"], sp, r_mask)["wape"] for sp in sd_preds]
                meta[f"{r_name}_D_wape_std"] = round(float(np.std(wapes_sd)), 3)
        else:
            for r_name in masks.keys():
                meta[f"{r_name}_D_wape_std"] = 0.0

        m_mac, m_mic = evaluate_model_macro_and_micro(m_id, pd_arr, data, ps_arr, meta=meta)
        macro_rows.append(m_mac)
        micro_rows.extend(m_mic)

    df_macro = pd.DataFrame(macro_rows)
    df_micro = pd.DataFrame(micro_rows)
    df_macro.to_csv(os.path.join(repo_root, "results/metrics/exp2_model_family_metrics.csv"), index=False)
    df_micro.to_csv(os.path.join(repo_root, "results/metrics/exp2_model_family_by_store.csv"), index=False)

    # Save test-period raw predictions table (28 days x 10 stores = 280 rows)
    train_end = data["train_end"]
    T = data["T"]
    N = data["N_stores"]
    pred_records = []
    for t in range(train_end, T):
        d_str = data["dates"][t].strftime("%Y-%m-%d")
        for i in range(N):
            s_id = data["store_ids"][i]
            c_val = data["cannibalization"][t, i]
            sp_val = data["spillover"][t, i]
            v_val = data["V_it"][t, i]
            if data["O_t"][t] == 0:
                reg_str = "Closed"
            elif sp_val > 5.0:
                reg_str = "Spillover"
            elif c_val > 5.0:
                reg_str = "Cannibalization"
            elif v_val < 100.0:
                reg_str = "Own_Stockout"
            else:
                reg_str = "Normal"

            row = {
                "ds": d_str,
                "store_id": s_id,
                "cluster": data["cluster_names"][i],
                "warehouse": f"W{data['w_assign'][i]+1}",
                "role": data["store_roles"][i],
                "is_open": int(data["O_t"][t]),
                "regime": reg_str,
                "discount": round(float(data["q_it"][t, i]), 2),
                "supply_cap": round(float(v_val), 1),
                "mu_self": round(float(data["mu_self"][t, i]), 2),
                "cannibalization": round(float(c_val), 2),
                "spillover_L": round(float(sp_val), 2),
                "mu_open": round(float(data["mu_open"][t, i]), 2),
                "D_sim": round(float(data["D_sim"][t, i]), 2),
                "y_sales": round(float(data["y_sales"][t, i]), 2),
            }
            for m_id, _, _, pd_arr, _, _, _, _, _, _ in model_entries:
                row[f"pred_{m_id}"] = round(float(pd_arr[t, i]), 2)
            pred_records.append(row)

    df_preds = pd.DataFrame(pred_records)
    df_preds.to_csv(os.path.join(repo_root, "results/predictions/synth_test_predictions_all_models.csv"), index=False)

    # Save plot-ready CSVs for Fig 4 (S5 & S6 time series), Fig 5 (Regime Ablation), and Fig 6 (Toolkit Tradeoff)
    df_preds[df_preds["store_id"].isin(["S5", "S6"])].to_csv(
        os.path.join(repo_root, "results/plot_data/plot_fig4_test_28d_s5_s6.csv"), index=False
    )
    df_macro.to_csv(os.path.join(repo_root, "results/plot_data/plot_fig5_ablation_regimes.csv"), index=False)
    df_macro.to_csv(os.path.join(repo_root, "results/plot_data/plot_fig6_toolkit_tradeoff.csv"), index=False)

    # ==========================================================================
    # 4. Experiment E5: Negative Control (Zero Coupling: cannib_beta=0, spillover_eta=0)
    # ==========================================================================
    data_neg = generate_retail_network_data(seed=42, cannib_beta=0.0, spillover_eta=0.0)
    feats_neg = build_all_feature_sets(data_neg)
    mask_neg_open = (np.arange(T)[:, None] >= train_end) & (data_neg["O_t"][:, None] == 1.0)
    mask_neg_open = np.broadcast_to(mask_neg_open, (T, N))

    p_neg_1b, _, _, _ = fit_predict_linear_spec("1B", feats_neg["L1_1B_Local_Uncensored"], data_neg)
    p_neg_l3, _, _, _ = fit_predict_linear_spec("L3", feats_neg["L3_2_Linear_1Hop_Local"], data_neg)
    p_neg_l4, _, _, _ = fit_predict_linear_spec("L4", feats_neg["L4_3_Analytical_2Hop_Graph"], data_neg)
    gbdt_neg = fit_predict_gbdt_multi_seed(feats_neg["Tree_GBDT_1Hop_18d"], data_neg, seeds=[42, 101, 202], engine="lgbm")
    xgb_neg = fit_predict_gbdt_multi_seed(feats_neg["Tree_GBDT_1Hop_18d"], data_neg, seeds=[42, 101, 202], engine="xgboost")
    xgb_27_neg = fit_predict_gbdt_multi_seed(feats_neg["Tree_GBDT_2Hop_27d"], data_neg, seeds=[42, 101, 202], engine="xgboost")
    gnn_neg = fit_predict_neural_gnn_multi_seed(data_neg, seeds=[42, 101, 202], hidden_dim=16, epochs=300)

    neg_rows = []
    for m_name, p_arr in [
        ("M1B_OLS_Uncens_7d", p_neg_1b),
        ("M2_OLS_1Hop_9d", p_neg_l3),
        ("M_GBDT_1Hop_Raw_18d", gbdt_neg["mean_pred_demand"]),
        ("M_XGBoost_1Hop_Raw_18d", xgb_neg["mean_pred_demand"]),
        ("M_XGBoost_2Hop_Graph_27d", xgb_27_neg["mean_pred_demand"]),
        ("M3_Neural_HeteroGNN_2Layer", gnn_neg["mean_pred_demand"]),
        ("M3_Analytical_2Hop_79d", p_neg_l4),
    ]:
        m_d = compute_metrics(data_neg["D_sim"], p_arr, mask_neg_open)
        m_mu = compute_metrics(data_neg["mu_open"], p_arr, mask_neg_open)
        neg_rows.append({
            "model_id": m_name,
            "scenario": "Negative_Control_Zero_Coupling",
            "D_wape": round(m_d["wape"], 3),
            "D_mae": round(m_d["mae"], 3),
            "D_bias": round(m_d["bias"], 3),
            "Mu_wape": round(m_mu["wape"], 3),
        })
    df_neg = pd.DataFrame(neg_rows)
    df_neg.to_csv(os.path.join(repo_root, "results/metrics/exp2_negative_control_metrics.csv"), index=False)

    # ==========================================================================
    # 5. Experiment E6: Coupling Strength Phase Sweep (beta_cannib x eta_spillover)
    # ==========================================================================
    sweep_rows = []
    for beta_c in [0.0, 40.0, 80.0, 125.0, 160.0]:
        for eta_s in [0.0, 0.25, 0.50, 0.85, 1.00]:
            d_sw = generate_retail_network_data(seed=42, cannib_beta=beta_c, spillover_eta=eta_s)
            f_sw = build_all_feature_sets(d_sw)
            m_sw = (np.arange(T)[:, None] >= train_end) & (d_sw["O_t"][:, None] == 1.0)
            m_sw = np.broadcast_to(m_sw, (T, N))
            p_sw_1b, _, _, _ = fit_predict_linear_spec("1B", f_sw["L1_1B_Local_Uncensored"], d_sw)
            p_sw_l3, _, _, _ = fit_predict_linear_spec("L3", f_sw["L3_2_Linear_1Hop_Local"], d_sw)
            p_sw_l4, _, _, _ = fit_predict_linear_spec("L4", f_sw["L4_3_Analytical_2Hop_Graph"], d_sw)
            w_1b = compute_metrics(d_sw["D_sim"], p_sw_1b, m_sw)["wape"]
            w_l3 = compute_metrics(d_sw["D_sim"], p_sw_l3, m_sw)["wape"]
            w_l4 = compute_metrics(d_sw["D_sim"], p_sw_l4, m_sw)["wape"]
            sweep_rows.append({
                "cannib_beta": beta_c,
                "spillover_eta": eta_s,
                "wape_1B_single_store": round(w_1b, 3),
                "wape_2_1hop_flat": round(w_l3, 3),
                "wape_3_2hop_graph": round(w_l4, 3),
                "graph_wape_gain_over_1B": round(w_1b - w_l4, 3),
            })
    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(os.path.join(repo_root, "results/metrics/exp2_coupling_phase_sweep.csv"), index=False)

    print("[Exp 2] Completed Cross-Model Family Benchmark, Negative Control, and Phase Sweep.")
    return {"macro": df_macro, "micro": df_micro, "preds": df_preds, "neg": df_neg, "sweep": df_sweep}


if __name__ == "__main__":
    run_exp2(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
