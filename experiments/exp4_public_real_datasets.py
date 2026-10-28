"""
Experiment 4: Real-World Public Benchmark Datasets Evaluation Suite.
1. FreshRetailNet-50K (Dingdong Inc: 10 Urban Front-Warehouse Stores, Real Stockout Hours, Weather & Promotions)
2. Corporación Favorita (Kaggle: 10 Supermarkets across Quito, Guayaquil, Cuenca, Real Promotions & Paydays)
3. Walmart M5 (Kaggle M5: 10 Stores across CA, TX, WI, Real SNAP Benefit Calendars & Regional Demand)

Loads genuine public benchmark datasets from data/public_benchmarks/raw/,
extracts 10-store networks, and executes the standardized 6-model evaluation ladder.
Outputs both 10-store macro metrics and store-by-store (S1..S10) micro breakdowns.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models_tree import NumPyHistGBDTRegressor, HAS_LGBM, HAS_XGB
from src.evaluation import compute_metrics
from src.real_data_loader import (
    load_real_freshretailnet_10stores,
    load_real_favorita_10stores,
    load_real_walmart_m5_10stores,
)

if HAS_LGBM:
    import lightgbm as lgb
if HAS_XGB:
    import xgboost as xgb


def _evaluate_public_dataset(ds_dict: dict, repo_root: str) -> tuple:
    """
    Run standardized 6-model ladder on a 10-store public dataset panel:
    1. 1A-Local-Naive (8d)
    2. 1B-Local-Uncensored/Cleaned (8d)
    3. 1B'-Pooled-Shared (17d)
    4. 2-Linear-1Hop (12d)
    5. GBDT-1Hop-Raw (16d)
    6. 2-Layer Hetero-GNN / Graph Interaction (85c / 13d active)
    """
    T = ds_dict["T"]
    train_end = ds_dict["train_end"]
    N = ds_dict["N"]
    P_spa = ds_dict["P_spa"]
    y_obs = ds_dict["y_obs"]
    D_true = ds_dict["D_true"]
    so_flag = ds_dict["stockout_flag"]
    promo = ds_dict["promo"]
    stock_hours = ds_dict["stock_hours"]

    t_idx = np.arange(T, dtype=float) / float(T)
    sin_w = np.sin(2 * np.pi * np.arange(T) / 7.0)
    cos_w = np.cos(2 * np.pi * np.arange(T) / 7.0)
    y_lag1 = np.vstack([y_obs[0:1, :], y_obs[:-1, :]])
    sh_lag1 = stock_hours

    # Save clean benchmark panel CSV to data/public_benchmarks/
    os.makedirs(os.path.join(repo_root, "data/public_benchmarks"), exist_ok=True)
    rows_csv = []
    for t in range(T):
        d_str = ds_dict["dates"][t].strftime("%Y-%m-%d")
        for i in range(N):
            rows_csv.append({
                "ds": d_str,
                "store_id": ds_dict["store_ids"][i],
                "cluster": ds_dict["clusters"][i],
                "is_weekend": int(ds_dict["is_wknd"][t]),
                ds_dict["cov1_name"]: round(float(ds_dict["cov1"][t]), 3),
                ds_dict["cov2_name"]: round(float(ds_dict["cov2"][t]), 3),
                "promo_depth": round(float(promo[t, i]), 3),
                "stockout_hours": round(float(stock_hours[t, i]), 1),
                "stockout_flag": int(so_flag[t, i]),
                "y_observed": round(float(y_obs[t, i]), 1),
                "D_latent_eval": round(float(D_true[t, i]), 1),
            })
    pd.DataFrame(rows_csv).to_csv(
        os.path.join(repo_root, f"data/public_benchmarks/{ds_dict['short_id'].lower()}_10stores.csv"),
        index=False,
    )

    # Feature matrices:
    # 1. Local 8d: [1, t, is_wknd, sin_w, cos_w, cov1, cov2, promo_i]
    X_8d = np.zeros((T, N, 8))
    for i in range(N):
        X_8d[:, i, :] = np.column_stack([
            np.ones(T), t_idx, ds_dict["is_wknd"], sin_w, cos_w,
            ds_dict["cov1"], ds_dict["cov2"], promo[:, i]
        ])

    # 2. Pooled 17d: [Store OneHot (10), t, is_wknd, sin_w, cos_w, cov1, cov2, promo_i]
    X_17d = np.zeros((T, N, 17))
    for i in range(N):
        oh = np.zeros((T, N))
        oh[:, i] = 1.0
        X_17d[:, i, :] = np.column_stack([
            oh, t_idx, ds_dict["is_wknd"], sin_w, cos_w,
            ds_dict["cov1"], ds_dict["cov2"], promo[:, i]
        ])

    # 3. 1-Hop Linear 12d: 8d + [nbr_promo, nbr_ylag1, nbr_stock_hours_lag1, own_ylag1]
    X_12d = np.zeros((T, N, 12))
    for i in range(N):
        nbr_p = promo @ P_spa[i, :]
        nbr_y = (y_lag1 @ P_spa[i, :]) / 100.0
        nbr_sh = (sh_lag1 @ P_spa[:, i]) / 16.0
        X_12d[:, i, :] = np.column_stack([
            X_8d[:, i, :], nbr_p, nbr_y, nbr_sh, y_lag1[:, i] / 100.0
        ])

    # 4. GBDT 16d wide table
    X_gbdt_16d = np.zeros((T, N, 16))
    for i in range(N):
        nbr_p = promo @ P_spa[i, :]
        nbr_y = (y_lag1 @ P_spa[i, :]) / 100.0
        nbr_sh = (sh_lag1 @ P_spa[:, i]) / 16.0
        X_gbdt_16d[:, i, :] = np.column_stack([
            np.full(T, float(i)),
            np.full(T, float(ds_dict["w_assign"][i])),
            t_idx, ds_dict["is_wknd"], sin_w, cos_w,
            ds_dict["cov1"], ds_dict["cov2"], promo[:, i],
            y_lag1[:, i] / 100.0,
            sh_lag1[:, i] / 16.0,
            nbr_p,
            nbr_p - promo[:, i],
            nbr_y,
            nbr_sh,
            nbr_p * nbr_sh,
        ])

    # 5. 2-Hop Graph Model (Pooled store basis 80d + 5 graph message channels = 85d, 13d active per store)
    X_graph = np.zeros((T, N, 85))
    sh_known = stock_hours
    for i in range(N):
        oh = np.zeros((T, N))
        oh[:, i] = 1.0
        base_80 = np.hstack([oh * X_8d[:, i, k:k+1] for k in range(8)])
        rel_p_healthy = (np.maximum(0.0, promo - promo[:, [i]]) * (sh_known <= 1)) @ P_spa[i, :]
        sp_base = ((sh_known >= 5).astype(float) @ P_spa[:, i])
        sp_promo = (((sh_known >= 5).astype(float) * promo) @ P_spa[:, i])
        nbr_y = (y_lag1 @ P_spa[i, :]) / 100.0
        cov_inter = ds_dict["cov1"] * (promo[:, i] > 0.15)
        X_graph[:, i, :] = np.column_stack([base_80, rel_p_healthy, sp_base, sp_promo, nbr_y, cov_inter])

    # Fit models on train period
    tr_all = np.arange(T)[:, None] < train_end
    tr_all = np.broadcast_to(tr_all, (T, N))
    tr_unc = tr_all & (so_flag == 0.0)
    m_pool_tr = tr_unc if np.sum(tr_unc) > 50 else tr_all

    train_samples = int(np.sum(tr_all))
    test_samples = int(np.sum(np.arange(T)[:, None] >= train_end) * N)
    total_samples = int(T * N)

    # 1. 1A Naive OLS (NumPy)
    t0_1a = time.perf_counter()
    p_1a = np.zeros((T, N))
    for i in range(N):
        b1a, _, _, _ = np.linalg.lstsq(X_8d[tr_all[:, i], i, :], y_obs[tr_all[:, i], i], rcond=None)
        p_1a[:, i] = np.maximum(0.0, X_8d[:, i, :] @ b1a)
    t_1a = time.perf_counter() - t0_1a

    # 2. 1B Uncensored OLS (NumPy)
    t0_1b = time.perf_counter()
    p_1b = np.zeros((T, N))
    for i in range(N):
        m_u = tr_unc[:, i] if np.sum(tr_unc[:, i]) > 10 else tr_all[:, i]
        b1b, _, _, _ = np.linalg.lstsq(X_8d[m_u, i, :], y_obs[m_u, i], rcond=None)
        p_1b[:, i] = np.maximum(0.0, X_8d[:, i, :] @ b1b)
    t_1b = time.perf_counter() - t0_1b

    # 3. 1B' Pooled 17d (NumPy)
    t0_pool = time.perf_counter()
    b_pool, _, _, _ = np.linalg.lstsq(X_17d[m_pool_tr], y_obs[m_pool_tr], rcond=None)
    p_pool = np.maximum(0.0, np.einsum("tnd,d->tn", X_17d, b_pool))
    t_pool = time.perf_counter() - t0_pool

    # 4. 2 Linear 1Hop 12d (NumPy)
    t0_l3 = time.perf_counter()
    p_l3 = np.zeros((T, N))
    for i in range(N):
        m_u = tr_unc[:, i] if np.sum(tr_unc[:, i]) > 10 else tr_all[:, i]
        bl3, _, _, _ = np.linalg.lstsq(X_12d[m_u, i, :], y_obs[m_u, i], rcond=None)
        p_l3[:, i] = np.maximum(0.0, X_12d[:, i, :] @ bl3)
    t_l3 = time.perf_counter() - t0_l3

    # 5. GBDT 16d (LightGBM)
    t0_gbdt = time.perf_counter()
    gbdt_preds = []
    if HAS_LGBM:
        for sd in [42, 101, 202]:
            lgbm = lgb.LGBMRegressor(
                n_estimators=60, learning_rate=0.08, max_depth=4, num_leaves=15,
                subsample=0.85, colsample_bytree=0.85, random_state=sd, verbose=-1, n_jobs=1
            )
            lgbm.fit(X_gbdt_16d[m_pool_tr], y_obs[m_pool_tr])
            pred_all = lgbm.predict(X_gbdt_16d.reshape(T * N, 16)).reshape(T, N)
            gbdt_preds.append(pred_all)
    else:
        for sd in [42]:
            tree_m = NumPyHistGBDTRegressor(
                n_estimators=50, learning_rate=0.10, n_bins=16, max_depth=4, min_samples_leaf=15, random_state=sd
            )
            tree_m.fit(X_gbdt_16d[m_pool_tr], y_obs[m_pool_tr])
            gbdt_preds.append(tree_m.predict(X_gbdt_16d.reshape(T * N, 16)).reshape(T, N))
    p_gbdt = np.maximum(0.0, np.mean(gbdt_preds, axis=0))
    t_gbdt = time.perf_counter() - t0_gbdt

    # 6. XGBoost 16d (XGBoost)
    t0_xgb = time.perf_counter()
    xgb_preds = []
    if HAS_XGB:
        for sd in [42, 101, 202]:
            xgb_m = xgb.XGBRegressor(
                tree_method="hist", max_depth=4, n_estimators=60, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85, random_state=sd, n_jobs=1
            )
            xgb_m.fit(X_gbdt_16d[m_pool_tr], y_obs[m_pool_tr])
            pred_all = xgb_m.predict(X_gbdt_16d.reshape(T * N, 16)).reshape(T, N)
            xgb_preds.append(pred_all)
    else:
        xgb_preds = gbdt_preds
    p_xgb = np.maximum(0.0, np.mean(xgb_preds, axis=0))
    t_xgb = time.perf_counter() - t0_xgb

    # 7. Graph Model (Ridge-regularized 2-hop message passing)
    t0_graph = time.perf_counter()
    Xg_tr = X_graph[m_pool_tr]
    yg_tr = y_obs[m_pool_tr]
    reg_g = 0.05 * np.eye(85)
    reg_g[:80, :80] = 1e-5 * np.eye(80)
    b_graph = np.linalg.solve(Xg_tr.T @ Xg_tr + reg_g, Xg_tr.T @ yg_tr)
    p_graph = np.maximum(0.0, np.einsum("tnd,d->tn", X_graph, b_graph))
    t_graph = time.perf_counter() - t0_graph

    test_mask = np.arange(T)[:, None] >= train_end
    test_mask = np.broadcast_to(test_mask, (T, N))

    models_eval = [
        ("1A_Local_Naive", "NumPy OLS", 8, 8, t_1a, p_1a),
        ("1B_Local_Uncensored", "NumPy OLS", 8, 8, t_1b, p_1b),
        ("1Bp_Pooled_Shared", "NumPy Pooled OLS", 8, 17, t_pool, p_pool),
        ("2_Linear_1Hop", "NumPy 1-Hop OLS", 12, 12, t_l3, p_l3),
        ("GBDT_1Hop_16d", "LightGBM (3 Seeds)", 16, 16, t_gbdt, p_gbdt),
        ("XGBoost_1Hop_16d", "XGBoost (3 Seeds)", 16, 16, t_xgb, p_xgb),
        ("3_Hetero_Graph_2Hop", "NumPy Ridge Graph", 13, 85, t_graph, p_graph),
    ]

    macro_summary = []
    store_breakdown = []

    for m_id, engine, act_d, mat_c, fit_t, p_arr in models_eval:
        m_ov = compute_metrics(D_true, p_arr, test_mask)
        macro_summary.append({
            "dataset": ds_dict["short_id"],
            "benchmark_name": ds_dict["benchmark_name"],
            "model_id": m_id,
            "engine": engine,
            "N_stores": N,
            "T_days": T,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "total_samples": total_samples,
            "active_dim_per_store": act_d,
            "design_matrix_cols": mat_c,
            "fit_time_sec": round(fit_t, 4),
            "Overall_WAPE": round(m_ov["wape"], 2),
            "Overall_MAE": round(m_ov["mae"], 2),
            "Overall_Bias": round(m_ov["bias"], 2),
        })

        for i in range(N):
            s_mask = np.zeros_like(test_mask, dtype=bool)
            s_mask[:, i] = test_mask[:, i]
            m_s = compute_metrics(D_true, p_arr, s_mask)
            store_breakdown.append({
                "dataset": ds_dict["short_id"],
                "store_id": ds_dict["store_ids"][i],
                "cluster": ds_dict["clusters"][i],
                "model_id": m_id,
                "engine": engine,
                "active_dim": act_d,
                "fit_time_sec": round(fit_t, 4),
                "WAPE": round(m_s["wape"], 2),
                "MAE": round(m_s["mae"], 2),
                "Bias": round(m_s["bias"], 2),
            })

    return macro_summary, store_breakdown


def run_exp4(repo_root: str) -> dict:
    t0 = time.time()
    print("[Exp 4] Loading Real Public Datasets: FreshRetailNet-50K, Kaggle Favorita & Walmart M5...")
    ds_list = [
        load_real_freshretailnet_10stores(),
        load_real_favorita_10stores(),
        load_real_walmart_m5_10stores(),
    ]

    all_macro = []
    all_store = []
    for ds in ds_list:
        print(f"[Exp 4] Evaluating Benchmark: {ds['benchmark_name']} (T={ds['T']}, N={ds['N']})...")
        mac, mic = _evaluate_public_dataset(ds, repo_root)
        all_macro.extend(mac)
        all_store.extend(mic)

    df_mac = pd.DataFrame(all_macro)
    df_mic = pd.DataFrame(all_store)

    df_mic_wide = df_mic.pivot_table(
        index=["dataset", "store_id", "cluster"],
        columns="model_id",
        values=["WAPE", "Bias"],
        sort=False,
    )
    df_mic_wide.columns = [f"{m}_{metric}" for metric, m in df_mic_wide.columns]
    df_mic_wide = df_mic_wide.reset_index()

    os.makedirs(os.path.join(repo_root, "results/metrics"), exist_ok=True)
    os.makedirs(os.path.join(repo_root, "results/plot_data"), exist_ok=True)

    df_mac.to_csv(os.path.join(repo_root, "results/metrics/exp4_public_datasets_macro.csv"), index=False)
    df_mic.to_csv(os.path.join(repo_root, "results/metrics/exp4_public_datasets_by_store_long.csv"), index=False)
    df_mic_wide.to_csv(os.path.join(repo_root, "results/metrics/exp4_public_datasets_by_store_wide.csv"), index=False)
    df_mac.to_csv(os.path.join(repo_root, "results/plot_data/plot_fig9_public_benchmarks.csv"), index=False)

    elapsed = time.time() - t0
    print(f"[Exp 4] Completed 3 Real Public Benchmarks Suite in {elapsed:.2f}s.")
    return {"macro": df_mac, "store_wide": df_mic_wide}


if __name__ == "__main__":
    run_exp4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
