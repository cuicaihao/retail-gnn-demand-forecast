"""
Experiment 3: Store-by-Store (S1..S10) x Regime Breakdown
+ Signal Distortion Sensitivity (A1..A3) + Fault Injection Blast Radius (B1..B3).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_generator import generate_retail_network_data
from src.feature_builders import build_all_feature_sets
from src.models_linear import fit_predict_linear_spec
from src.models_tree import fit_predict_gbdt_multi_seed
from src.models_gnn import fit_predict_neural_gnn_multi_seed
from src.evaluation import get_test_regime_masks, compute_metrics


def run_exp3(repo_root: str) -> dict:
    data = generate_retail_network_data(seed=42)
    feats = build_all_feature_sets(data)
    masks = get_test_regime_masks(data)

    # Fit key models
    p_1a, _, _, _ = fit_predict_linear_spec("1A", feats["L0_1A_Local_Naive"], data)
    p_1b, _, _, _ = fit_predict_linear_spec("1B", feats["L1_1B_Local_Uncensored"], data)
    p_l2a, _, _, _ = fit_predict_linear_spec("L2a", feats["L2a_1Bp_Pooled_Shared"], data)
    p_l3, _, _, _ = fit_predict_linear_spec("L3", feats["L3_2_Linear_1Hop_Local"], data)
    p_l4, _, _, _ = fit_predict_linear_spec("L4", feats["L4_3_Analytical_2Hop_Graph"], data)
    pred_csv = os.path.join(repo_root, "results/predictions/synth_test_predictions_all_models.csv")
    df_p = pd.read_csv(pred_csv)
    T_tot, N_st = data["T"], data["N_stores"]
    tr_end = data["train_end"]
    def _load_p(col_name):
        arr = np.zeros((T_tot, N_st))
        piv = df_p.pivot(index="ds", columns="store_id", values=col_name)[data["store_ids"]].values
        arr[tr_end:, :] = piv
        return arr
    gbdt_11 = _load_p("pred_M_GBDT_Base_11d")
    xgb_11 = _load_p("pred_M_XGBoost_Base_11d")
    gbdt_18 = _load_p("pred_M_GBDT_1Hop_Raw_18d")
    xgb_18 = _load_p("pred_M_XGBoost_1Hop_Raw_18d")
    gbdt_27 = _load_p("pred_M_GBDT_2Hop_Graph_27d")
    xgb_27 = _load_p("pred_M_XGBoost_2Hop_Graph_27d")
    prop_base = _load_p("pred_M_Prophet_Base")
    prop_1h = _load_p("pred_M_Prophet_1Hop")
    prop_2h = _load_p("pred_M_Prophet_2Hop")
    gnn_neu = _load_p("pred_M3_Neural_HeteroGNN_2Layer")

    models_map = [
        ("1A_Naive_7d", p_1a),
        ("1B_Uncens_7d", p_1b),
        ("1Bp_Pooled_16d", p_l2a),
        ("2_Linear_1Hop_9d", p_l3),
        ("Prophet_Base", prop_base),
        ("Prophet_1Hop", prop_1h),
        ("Prophet_2Hop", prop_2h),
        ("LightGBM_Base_11d", gbdt_11),
        ("LightGBM_1Hop_18d", gbdt_18),
        ("LightGBM_2Hop_27d", gbdt_27),
        ("XGBoost_Base_11d", xgb_11),
        ("XGBoost_1Hop_18d", xgb_18),
        ("XGBoost_2Hop_27d", xgb_27),
        ("3_Neural_GNN_2L", gnn_neu),
        ("3_Analytical_GNN_79d", p_l4),
    ]

    N = data["N_stores"]
    D_sim = data["D_sim"]

    # 1. Wide table: 10 Stores Overall Test Comparison (S1..S10)
    store_overall_rows = []
    # 2. Long & Wide table: 10 Stores x 4 Regimes (Overall, Normal, Cannibalization, Spillover)
    store_regime_rows = []

    for i in range(N):
        s_id = data["store_ids"][i]
        c_name = data["cluster_names"][i]
        w_id = f"W{data['w_assign'][i]+1}"
        role = data["store_roles"][i]

        n_open = int(np.sum(masks["Overall"][:, i]))
        n_norm = int(np.sum(masks["Normal"][:, i]))
        n_cann = int(np.sum(masks["Cannibalization"][:, i]))
        n_spil = int(np.sum(masks["Spillover"][:, i]))
        n_stockout = int(np.sum((masks["Overall"][:, i]) & (data["stockout_flag"][:, i] == 1.0)))

        ov_row = {
            "store_id": s_id,
            "cluster": c_name,
            "warehouse": w_id,
            "role": role,
            "open_days": n_open,
            "normal_days": n_norm,
            "cannib_days": n_cann,
            "spillover_days": n_spil,
            "own_stockout_days": n_stockout,
            "mean_latent_demand": round(float(np.mean(D_sim[masks["Overall"][:, i], i])), 1),
        }

        for r_name, r_mask in masks.items():
            s_mask = np.zeros_like(r_mask, dtype=bool)
            s_mask[:, i] = r_mask[:, i]
            n_r = int(np.sum(s_mask))

            reg_row = {
                "store_id": s_id,
                "cluster": c_name,
                "warehouse": w_id,
                "role": role,
                "regime": r_name,
                "n_days": n_r,
            }
            for m_code, p_arr in models_map:
                m_res = compute_metrics(D_sim, p_arr, s_mask)
                w_val = round(m_res["wape"], 2) if not np.isnan(m_res["wape"]) else np.nan
                b_val = round(m_res["bias"], 2) if not np.isnan(m_res["bias"]) else np.nan
                reg_row[f"{m_code}_wape"] = w_val
                reg_row[f"{m_code}_bias"] = b_val
                if r_name == "Overall":
                    ov_row[f"{m_code}_wape"] = w_val
                    ov_row[f"{m_code}_bias"] = b_val
            store_regime_rows.append(reg_row)

        store_overall_rows.append(ov_row)

    df_store_ov = pd.DataFrame(store_overall_rows)
    df_store_reg = pd.DataFrame(store_regime_rows)

    df_store_ov.to_csv(os.path.join(repo_root, "results/metrics/exp3_store_overall_comparison.csv"), index=False)
    df_store_reg.to_csv(os.path.join(repo_root, "results/metrics/exp3_store_by_regime_matrix.csv"), index=False)
    df_store_reg.to_csv(os.path.join(repo_root, "results/plot_data/plot_fig7_store_regime_heatmap.csv"), index=False)

    # ==========================================================================
    # 3. Experiment A: Signal Distortion Curves (Missing Stockout Flags & Delayed Warehouse State)
    # ==========================================================================
    rng = np.random.RandomState(42)
    T = data["T"]
    distortion_rows = []

    # A1: Stockout flag missing rate (false negatives in stockout_flag during training)
    for miss_rate in [0.0, 0.15, 0.30, 0.50, 0.75, 1.00]:
        so_corrupt = data["stockout_flag"].copy()
        if miss_rate > 0:
            drop_mask = rng.rand(T, N) < miss_rate
            so_corrupt = np.where(drop_mask, 0.0, so_corrupt)
        p1b_c, _, _, _ = fit_predict_linear_spec("1B", feats["L1_1B_Local_Uncensored"], data, stockout_flag_override=so_corrupt)
        pl3_c, _, _, _ = fit_predict_linear_spec("L3", feats["L3_2_Linear_1Hop_Local"], data, stockout_flag_override=so_corrupt)
        pl4_c, _, _, _ = fit_predict_linear_spec("L4", feats["L4_3_Analytical_2Hop_Graph"], data, stockout_flag_override=so_corrupt)

        for r_name in ["Overall", "Spillover"]:
            distortion_rows.append({
                "distortion_type": "Stockout_Miss_Rate",
                "distortion_level": miss_rate,
                "regime": r_name,
                "1B_Uncens_wape": round(compute_metrics(D_sim, p1b_c, masks[r_name])["wape"], 2),
                "1B_Uncens_bias": round(compute_metrics(D_sim, p1b_c, masks[r_name])["bias"], 2),
                "2_1Hop_wape": round(compute_metrics(D_sim, pl3_c, masks[r_name])["wape"], 2),
                "3_Graph_wape": round(compute_metrics(D_sim, pl4_c, masks[r_name])["wape"], 2),
                "3_Graph_bias": round(compute_metrics(D_sim, pl4_c, masks[r_name])["bias"], 2),
            })

    # A2: Warehouse state reporting delay (extra lag of 0, 1, 2, 3 days)
    for extra_delay in [0, 1, 2, 3]:
        nw_delayed = np.roll(data["node_w_state"], extra_delay, axis=0)
        if extra_delay > 0:
            nw_delayed[:extra_delay, :] = 0
        f_del = build_all_feature_sets(data, node_w_state_override=nw_delayed)
        pl3_d, _, _, _ = fit_predict_linear_spec("L3", f_del["L3_2_Linear_1Hop_Local"], data)
        pl4_d, _, _, _ = fit_predict_linear_spec("L4", f_del["L4_3_Analytical_2Hop_Graph"], data)
        for r_name in ["Overall", "Spillover"]:
            distortion_rows.append({
                "distortion_type": "Warehouse_State_Extra_Delay_Days",
                "distortion_level": float(extra_delay),
                "regime": r_name,
                "1B_Uncens_wape": round(compute_metrics(D_sim, p_1b, masks[r_name])["wape"], 2),
                "1B_Uncens_bias": round(compute_metrics(D_sim, p_1b, masks[r_name])["bias"], 2),
                "2_1Hop_wape": round(compute_metrics(D_sim, pl3_d, masks[r_name])["wape"], 2),
                "3_Graph_wape": round(compute_metrics(D_sim, pl4_d, masks[r_name])["wape"], 2),
                "3_Graph_bias": round(compute_metrics(D_sim, pl4_d, masks[r_name])["bias"], 2),
            })

    df_distort = pd.DataFrame(distortion_rows)
    df_distort.to_csv(os.path.join(repo_root, "results/metrics/exp3_signal_distortion_curve.csv"), index=False)

    # ==========================================================================
    # 4. Experiment B: Fault Injection & Blast Radius (Bridge S5/S6 vs. Edge S10)
    # ==========================================================================
    # Inject a corrupted promotion/lag-sales spike at test days 1072..1076 on fault_store
    # Compare blast radius on 0-hop (fault store), 1-hop neighbors, and unlinked stores (>1-hop)
    _, _, _, beta_l4 = fit_predict_linear_spec("L4", feats["L4_3_Analytical_2Hop_Graph"], data)
    blast_rows = []
    for fault_idx, fault_label in [(4, "S5_Central_W1_Source"), (5, "S6_Central_Bridge"), (9, "S10_West_Edge")]:
        data_corrupt = dict(data)
        q_corrupt = data["q_it"].copy()
        y_corrupt = data["y_sales"].copy()
        # Corrupt test window days 1072..1077 with a 10x anomalous input spike
        q_corrupt[1072:1077, fault_idx] = 3.50  # 10x normal max discount (data corruption)
        y_corrupt[1071:1076, fault_idx] = 950.0  # 10x normal sales spike
        data_corrupt["q_it"] = q_corrupt
        data_corrupt["y_sales"] = y_corrupt

        f_corrupt = build_all_feature_sets(data_corrupt)
        # Predict using pre-trained coefficients on corrupted test inputs
        pred_l4_corrupt = np.zeros((T, N))
        for i in range(N):
            pred_l4_corrupt[:, i] = data["O_t"] * np.maximum(0.0, f_corrupt["L4_3_Analytical_2Hop_Graph"]["X"][:, i, :] @ beta_l4)

        # Circuit breaker: if a neighbor's input deviates > 3 sigma from historical range, zero its outgoing edges & fallback to 1B
        P_cb = data["P_spatial"].copy()
        P_cb[fault_idx, :] = 0.0
        P_cb[:, fault_idx] = 0.0
        f_cb = build_all_feature_sets(data_corrupt, P_spa_override=P_cb)
        pred_l4_cb = np.zeros((T, N))
        for i in range(N):
            if i == fault_idx:
                pred_l4_cb[:, i] = p_1b[:, i]
            else:
                pred_l4_cb[:, i] = data["O_t"] * np.maximum(0.0, f_cb["L4_3_Analytical_2Hop_Graph"]["X"][:, i, :] @ beta_l4)

        for target_i in range(N):
            if target_i == fault_idx:
                hop_group = "0_Hop_Self"
            elif data["A_spatial"][fault_idx, target_i] > 0:
                hop_group = "1_Hop_Neighbor"
            else:
                hop_group = "Unlinked_Other_Cluster"

            s_mask = np.zeros_like(masks["Overall"], dtype=bool)
            s_mask[:, target_i] = masks["Overall"][:, target_i]

            w_clean = compute_metrics(D_sim, p_l4, s_mask)["wape"]
            w_corrupt = compute_metrics(D_sim, pred_l4_corrupt, s_mask)["wape"]
            w_cb = compute_metrics(D_sim, pred_l4_cb, s_mask)["wape"]
            w_1b_clean = compute_metrics(D_sim, p_1b, s_mask)["wape"]

            blast_rows.append({
                "fault_source": fault_label,
                "target_store": data["store_ids"][target_i],
                "target_cluster": data["cluster_names"][target_i],
                "hop_distance": hop_group,
                "wape_1B_isolated": round(w_1b_clean, 2),
                "wape_3_graph_clean": round(w_clean, 2),
                "wape_3_graph_corrupted": round(w_corrupt, 2),
                "wape_delta_no_breaker": round(w_corrupt - w_clean, 2),
                "wape_3_graph_with_circuit_breaker": round(w_cb, 2),
            })

    df_blast = pd.DataFrame(blast_rows)
    df_blast.to_csv(os.path.join(repo_root, "results/metrics/exp3_fault_injection_blast_radius.csv"), index=False)

    print("[Exp 3] Completed Store-by-Store x Regime Breakdown, Distortion Curves, and Blast Radius.")
    return {"store_ov": df_store_ov, "store_reg": df_store_reg, "distort": df_distort, "blast": df_blast}


if __name__ == "__main__":
    run_exp3(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
