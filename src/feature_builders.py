"""
Explicit Feature Builders for All Experimental Ladders (L0..L4, GBDT 11d/18d/27d, Graph Tensors).
Tracks exact feature dimensions, column names, and active per-store dimensions.
"""
import numpy as np
from typing import Dict, Any, List, Tuple


def build_all_feature_sets(
    data: Dict[str, Any],
    P_spa_override: np.ndarray = None,
    node_w_state_override: np.ndarray = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Construct all feature representations for:
    - L0 / L1: 7-dim single-store isolated features
    - L2a: 16-dim pooled shared-elasticity features (10 store intercepts + 6 shared covariates)
    - L2b: 70-dim pooled block-diagonal store-specific features (10 x 7, zero graph info)
    - L3_local: 9-dim per-store 1-hop flat features (11-dim for sales cap)
    - L3_pooled: 72-dim pooled 1-hop flat features (70 base + 2 shared 1-hop)
    - L4_analytical_2hop: 79-dim pooled 2-hop conditional interaction features (16 active per store)
    - GBDT_Base: 11-dim tabular features (no graph)
    - GBDT_1Hop_Raw: 18-dim raw 1-hop neighborhood wide table (no manual 2-hop product)
    - GBDT_2Hop_Graph: 27-dim full 1-hop + 2-hop graph-aggregated features
    """
    T = data["T"]
    N = data["N_stores"]
    t_idx = data["t_idx"]
    W_t = data["W_t"]
    H_t = data["H_t"]
    q_it = data["q_it"]
    y_sales = data["y_sales"]
    w_assign = data["w_assign"]
    P_spa = P_spa_override if P_spa_override is not None else data["P_spatial"]
    node_w = node_w_state_override if node_w_state_override is not None else data["node_w_state"]

    sin_y = np.sin(2.0 * np.pi * (t_idx - 15.0) / 365.25)
    cos_y = np.cos(2.0 * np.pi * (t_idx - 15.0) / 365.25)
    t_scaled = t_idx / 1000.0

    # Lagged sales (strictly causal: t-1 and rolling 7-day mean t-7..t-1)
    y_lag1 = np.vstack([y_sales[0:1, :], y_sales[:-1, :]])
    y_roll7 = np.zeros_like(y_sales)
    for t in range(T):
        start_r = max(0, t - 7)
        end_r = max(1, t)
        y_roll7[t] = y_sales[start_r:end_r].mean(axis=0)

    # 1. L0 & L1: Per-store 7-dim matrix X_local[T, N, 7]
    l0_cols = ["intercept", "t_scaled", "is_weekend", "sin_annual", "cos_annual", "is_holiday", "own_promo_q"]
    X_local_7d = np.zeros((T, N, 7))
    for i in range(N):
        X_local_7d[:, i, :] = np.column_stack([
            np.ones(T), t_scaled, W_t, sin_y, cos_y, H_t, q_it[:, i]
        ])

    # 2. L2a: Pooled Shared Elasticity (16 columns: 10 store one-hots + 6 shared covariates)
    l2a_cols = [f"store_oh_S{j+1}" for j in range(N)] + [
        "t_scaled", "is_weekend", "sin_annual", "cos_annual", "is_holiday", "own_promo_q"
    ]
    X_l2a_16d = np.zeros((T, N, 16))
    for i in range(N):
        oh = np.zeros((T, N))
        oh[:, i] = 1.0
        X_l2a_16d[:, i, :] = np.column_stack([
            oh, t_scaled, W_t, sin_y, cos_y, H_t, q_it[:, i]
        ])

    # 3. L2b: Pooled Block-Diagonal Store-Specific (70 columns: 10 stores x 7 features, zero graph)
    l2b_cols = []
    for j in range(N):
        l2b_cols.extend([f"S{j+1}_{c}" for c in l0_cols])
    X_l2b_70d = np.zeros((T, N, 70))
    for i in range(N):
        oh = np.zeros((T, N))
        oh[:, i] = 1.0
        X_l2b_70d[:, i, :] = np.hstack([
            oh,
            oh * t_scaled[:, None],
            oh * W_t[:, None],
            oh * sin_y[:, None],
            oh * cos_y[:, None],
            oh * H_t[:, None],
            oh * q_it[:, [i]],
        ])

    # 4. L3: 1-Hop Flat Neighborhood Features
    # Per-store 9-dim (or 11-dim with own warehouse lag1 state)
    l3_cols = l0_cols + ["nbr_weighted_promo_q", "nbr_weighted_lag1_sales"]
    l3_sales_cols = l3_cols + ["own_wh_congested_lag1", "own_wh_disrupted_lag1"]
    X_l3_9d = np.zeros((T, N, 9))
    X_l3_11d = np.zeros((T, N, 11))
    X_l3_pooled_72d = np.zeros((T, N, 72))
    for i in range(N):
        nbr_promo = q_it @ P_spa[i, :]
        nbr_ylag1 = (y_lag1 @ P_spa[i, :]) / 100.0
        w_c = (node_w[:, i] == 1).astype(float)
        w_d = (node_w[:, i] == 2).astype(float)
        X_l3_9d[:, i, :] = np.column_stack([X_local_7d[:, i, :], nbr_promo, nbr_ylag1])
        X_l3_11d[:, i, :] = np.column_stack([X_local_7d[:, i, :], nbr_promo, nbr_ylag1, w_c, w_d])
        X_l3_pooled_72d[:, i, :] = np.column_stack([X_l2b_70d[:, i, :], nbr_promo, nbr_ylag1])

    # 5. L4: 2-Hop Conditional Interaction Features (79 columns total; 16 active per store)
    msg_cols = [
        "msg_cannib_healthy_rel_promo",
        "msg_disrupt_base", "msg_disrupt_promo", "msg_disrupt_wknd", "msg_disrupt_sin", "msg_disrupt_hol",
        "msg_congest_base", "msg_congest_promo", "msg_congest_wknd",
    ]
    l4_cols = l2b_cols + msg_cols
    X_l4_79d = np.zeros((T, N, 79))
    X_msg_9d = np.zeros((T, N, 9))
    for i in range(N):
        rel_promo_healthy = np.maximum(0.0, q_it - q_it[:, [i]]) * (node_w == 0)
        msg_cannib = (rel_promo_healthy @ P_spa[i, :])[:, None]
        msg_disrupt_base = ((node_w == 2).astype(float) @ P_spa[:, i])[:, None]
        msg_disrupt_promo = ((q_it * (node_w == 2)) @ P_spa[:, i])[:, None]
        msg_disrupt_wknd = ((W_t[:, None] * (node_w == 2)) @ P_spa[:, i])[:, None]
        msg_disrupt_sin = ((sin_y[:, None] * (node_w == 2)) @ P_spa[:, i])[:, None]
        msg_disrupt_hol = ((H_t[:, None] * (node_w == 2)) @ P_spa[:, i])[:, None]
        msg_congest_base = ((node_w == 1).astype(float) @ P_spa[:, i])[:, None]
        msg_congest_promo = ((q_it * (node_w == 1)) @ P_spa[:, i])[:, None]
        msg_congest_wknd = ((W_t[:, None] * (node_w == 1)) @ P_spa[:, i])[:, None]

        m_i = np.hstack([
            msg_cannib,
            msg_disrupt_base, msg_disrupt_promo, msg_disrupt_wknd, msg_disrupt_sin, msg_disrupt_hol,
            msg_congest_base, msg_congest_promo, msg_congest_wknd,
        ])
        X_msg_9d[:, i, :] = m_i
        X_l4_79d[:, i, :] = np.hstack([X_l2b_70d[:, i, :], m_i])

    # 6. Tree / GBDT Feature Sets:
    # (a) GBDT_Base (11-dim: pure single-store & hierarchy IDs, zero neighbor/graph info)
    gbdt_base_cols = [
        "store_id_num", "cluster_id_num", "warehouse_id_num",
        "t_scaled", "is_weekend", "sin_annual", "cos_annual", "is_holiday", "own_promo_q",
        "own_sales_lag1", "own_sales_roll7",
    ]
    cluster_id_map = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    X_gbdt_11d = np.zeros((T, N, 11))
    for i in range(N):
        X_gbdt_11d[:, i, :] = np.column_stack([
            np.full(T, float(i)),
            np.full(T, float(cluster_id_map[i])),
            np.full(T, float(w_assign[i])),
            t_scaled, W_t, sin_y, cos_y, H_t, q_it[:, i],
            y_lag1[:, i] / 100.0, y_roll7[:, i] / 100.0,
        ])

    # (b) GBDT_1Hop_Raw (18-dim: adds raw 1-hop neighborhood summaries WITHOUT manual product interaction)
    gbdt_1hop_cols = gbdt_base_cols + [
        "own_wh_state_lag1",
        "nbr_weighted_promo_q",
        "nbr_max_promo_q",
        "nbr_promo_diff",
        "nbr_weighted_lag1_sales",
        "nbr_wh_disrupted_fraction",
        "nbr_wh_congested_fraction",
    ]
    X_gbdt_18d = np.zeros((T, N, 18))
    for i in range(N):
        nbr_mask = P_spa[i, :] > 0
        nbr_promo_w = q_it @ P_spa[i, :]
        nbr_promo_max = q_it[:, nbr_mask].max(axis=1) if nbr_mask.any() else np.zeros(T)
        nbr_promo_diff = nbr_promo_w - q_it[:, i]
        nbr_ylag1_w = (y_lag1 @ P_spa[i, :]) / 100.0
        nbr_disrupt_frac = (node_w == 2).astype(float) @ P_spa[:, i]
        nbr_congest_frac = (node_w == 1).astype(float) @ P_spa[:, i]
        X_gbdt_18d[:, i, :] = np.column_stack([
            X_gbdt_11d[:, i, :],
            node_w[:, i].astype(float),
            nbr_promo_w,
            nbr_promo_max,
            nbr_promo_diff,
            nbr_ylag1_w,
            nbr_disrupt_frac,
            nbr_congest_frac,
        ])

    # (c) GBDT_2Hop_Graph (27-dim: 18-dim + 9 explicit 2-hop message features)
    gbdt_2hop_cols = gbdt_1hop_cols + msg_cols
    X_gbdt_27d = np.concatenate([X_gbdt_18d, X_msg_9d], axis=-1)

    return {
        "L0_1A_Local_Naive": {
            "X": X_local_7d, "active_dim": 7, "matrix_cols": 7, "free_params": 70,
            "feature_names": l0_cols, "mode": "local_all_open"
        },
        "L1_1B_Local_Uncensored": {
            "X": X_local_7d, "active_dim": 7, "matrix_cols": 7, "free_params": 70,
            "feature_names": l0_cols, "mode": "local_uncensored"
        },
        "L2a_1Bp_Pooled_Shared": {
            "X": X_l2a_16d, "active_dim": 7, "matrix_cols": 16, "free_params": 16,
            "feature_names": l2a_cols, "mode": "pooled_uncensored"
        },
        "L2b_1Bp_Pooled_BlockDiag": {
            "X": X_l2b_70d, "active_dim": 7, "matrix_cols": 70, "free_params": 70,
            "feature_names": l2b_cols, "mode": "pooled_uncensored"
        },
        "L3_2_Linear_1Hop_Local": {
            "X": X_l3_9d, "X_sales": X_l3_11d, "active_dim": 9, "matrix_cols": 9, "free_params": 90,
            "feature_names": l3_cols, "mode": "local_uncensored"
        },
        "L3p_2_Linear_1Hop_Pooled": {
            "X": X_l3_pooled_72d, "active_dim": 9, "matrix_cols": 72, "free_params": 72,
            "feature_names": l2b_cols + ["nbr_weighted_promo_q", "nbr_weighted_lag1_sales"],
            "mode": "pooled_uncensored"
        },
        "L4_3_Analytical_2Hop_Graph": {
            "X": X_l4_79d, "active_dim": 16, "matrix_cols": 79, "free_params": 79,
            "feature_names": l4_cols, "mode": "pooled_uncensored"
        },
        "Tree_GBDT_Base_11d": {
            "X": X_gbdt_11d, "active_dim": 11, "matrix_cols": 11, "free_params": None,
            "feature_names": gbdt_base_cols, "mode": "pooled_uncensored"
        },
        "Tree_GBDT_1Hop_18d": {
            "X": X_gbdt_18d, "active_dim": 18, "matrix_cols": 18, "free_params": None,
            "feature_names": gbdt_1hop_cols, "mode": "pooled_uncensored"
        },
        "Tree_GBDT_2Hop_27d": {
            "X": X_gbdt_27d, "active_dim": 27, "matrix_cols": 27, "free_params": None,
            "feature_names": gbdt_2hop_cols, "mode": "pooled_uncensored"
        },
    }
