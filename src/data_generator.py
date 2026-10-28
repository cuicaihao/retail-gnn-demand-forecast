"""
Reproducible 4-Layer Coupled Retail Network Data Generator (DGP).
Supports:
- Standard benchmark (seed=42, cannib_beta=125.0, spillover_eta=0.85)
- Negative control (cannib_beta=0.0, spillover_eta=0.0)
- Coupling strength phase sweeps & multi-seed robustness runs
- Signal distortion (missing stockout flags, delayed/noisy warehouse states)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def generate_retail_network_data(
    seed: int = 42,
    cannib_beta: float = 125.0,
    spillover_eta: float = 0.85,
    noise_sigma: float = 4.5,
    spatial_sigma_d: float = 2.0,
    spatial_cutoff_km: float = 3.5,
    pin_test_regimes: bool = True,
) -> Dict[str, Any]:
    """Generate 1,096 days x 10 stores synthetic retail network data."""
    rng = np.random.RandomState(seed)

    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    T = len(dates)
    t_idx = np.arange(T, dtype=float)
    W_t = np.asarray(dates.dayofweek >= 5, dtype=float)

    vic_holidays_dates = [
        "2023-01-01", "2023-01-02", "2023-01-26", "2023-03-13", "2023-04-07",
        "2023-04-08", "2023-04-09", "2023-04-10", "2023-04-25", "2023-06-12",
        "2023-09-29", "2023-11-07", "2023-12-25", "2023-12-26",
        "2024-01-01", "2024-01-26", "2024-03-11", "2024-03-29", "2024-03-30",
        "2024-03-31", "2024-04-01", "2024-04-25", "2024-06-10", "2024-09-27",
        "2024-11-05", "2024-12-25", "2024-12-26",
        "2025-01-01", "2025-01-27", "2025-03-10", "2025-04-18", "2025-04-19",
        "2025-04-20", "2025-04-21", "2025-04-25", "2025-06-09", "2025-09-26",
        "2025-11-04", "2025-12-25", "2025-12-26",
    ]
    closed_holidays_dates = [
        "2023-04-07", "2023-12-25",
        "2024-03-29", "2024-12-25",
        "2025-04-18", "2025-12-25",
    ]
    H_t = np.asarray(dates.isin(pd.to_datetime(vic_holidays_dates)), dtype=float)
    is_closed_holiday = np.asarray(dates.isin(pd.to_datetime(closed_holidays_dates)), dtype=float)
    O_t = 1.0 - is_closed_holiday

    N_stores = 10
    N_warehouses = 2
    store_ids = [f"S{i+1}" for i in range(N_stores)]
    cluster_names = [
        "Cluster_A_East", "Cluster_A_East", "Cluster_A_East", "Cluster_A_East",
        "Cluster_B_Central", "Cluster_B_Central", "Cluster_B_Central",
        "Cluster_C_West", "Cluster_C_West", "Cluster_C_West",
    ]
    store_roles = [
        "Core_W1", "Core_W1", "Core_W1", "Bridge_W2",
        "Flank_W1", "Bridge_W2", "Flank_W1",
        "Control_W2", "Control_W2", "Control_W2",
    ]

    store_coords = np.array([
        [2.0, 8.0],   # S1 (Cluster A, W1)
        [3.0, 8.4],   # S2 (Cluster A, W1)
        [2.4, 7.0],   # S3 (Cluster A, W1)
        [3.5, 7.3],   # S4 (Cluster A, W2 Bridge)
        [7.5, 5.0],   # S5 (Cluster B, W1)
        [8.3, 5.5],   # S6 (Cluster B, W2 Bridge)
        [8.8, 4.6],   # S7 (Cluster B, W1)
        [14.0, 2.5],  # S8 (Cluster C, W2)
        [15.0, 2.9],  # S9 (Cluster C, W2)
        [14.4, 1.7],  # S10 (Cluster C, W2)
    ])
    warehouse_coords = np.array([[5.2, 9.8], [11.2, 1.2]])
    w_assign = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])

    dist_matrix = np.linalg.norm(store_coords[:, None, :] - store_coords[None, :, :], axis=-1)
    A_spatial = np.where(
        (dist_matrix > 0) & (dist_matrix <= spatial_cutoff_km),
        np.exp(-(dist_matrix ** 2) / (2.0 * (spatial_sigma_d ** 2))),
        0.0,
    )
    row_sums = A_spatial.sum(axis=1, keepdims=True)
    P_spatial = np.divide(A_spatial, row_sums, out=np.zeros_like(A_spatial), where=row_sums > 0)

    P_markov = np.array([
        [0.78, 0.15, 0.07],
        [0.42, 0.40, 0.18],
        [0.38, 0.32, 0.30],
    ])

    # Use np.random.seed(seed) for exact parity with original benchmark when seed==42
    if seed == 42:
        np.random.seed(42)
        W_state = np.zeros((T, N_warehouses), dtype=int)
        for w in range(N_warehouses):
            state = 0
            for t in range(T):
                W_state[t, w] = state
                state = np.random.choice([0, 1, 2], p=P_markov[state])
    else:
        W_state = np.zeros((T, N_warehouses), dtype=int)
        for w in range(N_warehouses):
            state = 0
            for t in range(T):
                W_state[t, w] = state
                state = rng.choice([0, 1, 2], p=P_markov[state])

    if pin_test_regimes:
        W_state[1068:, :] = 0
        W_state[1071:1076, 0] = 2  # W1 Disrupted 5 days
        W_state[1080:1084, 1] = 2  # W2 Disrupted 4 days
        W_state[1088:1092, 0] = 1  # W1 Congested 4 days

    base_demand = np.array([85, 82, 88, 84, 90, 86, 84, 80, 78, 83], dtype=float)
    trend_slope = np.array([0.015, 0.013, 0.016, 0.014, 0.017, 0.015, 0.015, 0.013, 0.012, 0.014])
    weekend_boost = np.array([16, 15, 17, 16, 19, 18, 17, 15, 14, 16], dtype=float)
    season_amp = np.array([14, 13, 15, 14, 16, 14, 14, 13, 12, 13], dtype=float)
    holiday_boost = np.array([28, 26, 29, 27, 31, 28, 28, 25, 23, 26], dtype=float)
    promo_elasticity = np.array([130, 125, 135, 130, 140, 135, 130, 125, 120, 128], dtype=float)

    phase_offsets = [0, 4, 8, 11, 1, 6, 10, 2, 7, 12]
    q_it = np.zeros((T, N_stores))
    for i in range(N_stores):
        cycle_day = (t_idx + phase_offsets[i]) % 14
        q_it[:, i] = np.where(cycle_day < 5, 0.20, 0.0)
        flash_day = (t_idx + phase_offsets[i]) % 21
        q_it[:, i] = np.where(flash_day < 3, 0.35, q_it[:, i])

    mu_self = np.zeros((T, N_stores))
    cannibalization = np.zeros((T, N_stores))
    spillover = np.zeros((T, N_stores))
    mu_open = np.zeros((T, N_stores))
    D_sim = np.zeros((T, N_stores))
    V_it = np.zeros((T, N_stores))
    y_sales = np.zeros((T, N_stores))
    stockout_flag = np.zeros((T, N_stores))

    cap_map = {0: 999.0, 1: 95.0, 2: 50.0}

    for t in range(T):
        w_prev = W_state[t - 1] if t > 0 else np.array([0, 0])
        for i in range(N_stores):
            V_it[t, i] = cap_map[w_prev[w_assign[i]]]

        sin_term = np.sin(2.0 * np.pi * (t - 15.0) / 365.25)
        for i in range(N_stores):
            mu_self[t, i] = (
                base_demand[i]
                + trend_slope[i] * t
                + weekend_boost[i] * W_t[t]
                + season_amp[i] * sin_term
                + holiday_boost[i] * H_t[t]
                + promo_elasticity[i] * q_it[t, i]
            )

        supply_healthy = (V_it[t] > 120.0).astype(float)
        for i in range(N_stores):
            cannib_sum = 0.0
            for j in range(N_stores):
                if A_spatial[i, j] > 0:
                    promo_diff = max(0.0, q_it[t, j] - q_it[t, i])
                    cannib_sum += P_spatial[i, j] * (cannib_beta * promo_diff) * supply_healthy[j]
            cannibalization[t, i] = cannib_sum

        unfulfilled_j = np.maximum(0.0, (mu_self[t] - cannibalization[t]) - V_it[t])
        for i in range(N_stores):
            spill_sum = 0.0
            for j in range(N_stores):
                if A_spatial[j, i] > 0:
                    spill_sum += spillover_eta * P_spatial[j, i] * unfulfilled_j[j]
            spillover[t, i] = spill_sum

        mu_open[t] = np.maximum(10.0, mu_self[t] - cannibalization[t] + spillover[t])
        if seed == 42:
            eps = np.random.normal(0.0, noise_sigma, size=N_stores)
        else:
            eps = rng.normal(0.0, noise_sigma, size=N_stores)
        D_sim[t] = np.maximum(0.0, mu_open[t] + eps)
        y_sales[t] = O_t[t] * np.round(np.minimum(D_sim[t], V_it[t]))
        stockout_flag[t] = ((D_sim[t] > V_it[t]) & (O_t[t] == 1.0)).astype(float)

    w_prev_all = np.vstack([[0, 0], W_state[:-1]])
    node_w_state = np.zeros((T, N_stores), dtype=int)
    for j in range(N_stores):
        node_w_state[:, j] = w_prev_all[:, w_assign[j]]

    return {
        "dates": dates,
        "T": T,
        "train_end": 1068,
        "t_idx": t_idx,
        "W_t": W_t,
        "H_t": H_t,
        "O_t": O_t,
        "N_stores": N_stores,
        "N_warehouses": N_warehouses,
        "store_ids": store_ids,
        "cluster_names": cluster_names,
        "store_roles": store_roles,
        "store_coords": store_coords,
        "warehouse_coords": warehouse_coords,
        "w_assign": w_assign,
        "dist_matrix": dist_matrix,
        "A_spatial": A_spatial,
        "P_spatial": P_spatial,
        "W_state": W_state,
        "node_w_state": node_w_state,
        "q_it": q_it,
        "mu_self": mu_self,
        "cannibalization": cannibalization,
        "spillover": spillover,
        "mu_open": mu_open,
        "D_sim": D_sim,
        "V_it": V_it,
        "y_sales": y_sales,
        "stockout_flag": stockout_flag,
    }
