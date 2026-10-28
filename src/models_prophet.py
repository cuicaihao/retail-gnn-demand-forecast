"""
Facebook Prophet Model Integration for Multi-Store Retail Demand Forecasting.
Provides the three-stage spatial information ladder aligned with the author's published blog design:
1. Prophet-Base (0-Hop: Native Weekly Fourier order=3 + Yearly Fourier order=5 + is_weekend + discount)
2. Prophet-1Hop (1-Hop: Base + neighbor discount, lag sales, stockout flag)
3. Prophet-2Hop (2-Hop: 1-Hop + 2-hop random walk diffusion, upstream warehouse state)

Feature Engineering Philosophy & Underlying Logic:
- Intrinsic Calendar Periodicity: Handled natively by continuous Fourier expansion (weekly period=7 fourier_order=3,
  yearly period=365.25 fourier_order=5) with Bayesian L2 shrinkage. We never feed artificial sin_year/cos_year columns.
- Business Step Jump: is_weekend is provided explicitly as an add_regressor() to capture the sharp non-smooth
  square-wave step jump between weekdays and weekends, and to ensure feature parity with tree/linear models.
- Independent Holiday Impulses: Configured natively via Prophet's holidays DataFrame with independent holidays_prior_scale.
- Spatial Signals: Injected as true exogenous regressors (add_regressor) representing external topological shocks.
"""
import time
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Suppress verbose prophet/cmdstanpy terminal logs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def _build_prophet_store_dfs(
    data: Dict[str, Any],
    store_idx: int,
    spatial_level: str = "base",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build clean train and future test DataFrames for store_idx,
    plus a unified retail holidays DataFrame.
    """
    T = data["T"]
    train_end = data["train_end"]
    dates = data["dates"]
    y_sales = data["y_sales"]
    O_t = data["O_t"]
    so_flag = data["stockout_flag"]
    disc = data["q_it"]
    P_spa = data["P_spatial"]

    t_arr = np.arange(T)
    d_str = [d.strftime("%Y-%m-%d") for d in dates]

    # 1. Base exogenous signals:
    # is_weekend provides the sharp weekend step-jump;
    # discount provides the endogenous business pricing driver.
    # (sin_year and cos_year are omitted as annual cycles are handled natively by Fourier yearly seasonality).
    df_all = pd.DataFrame({
        "ds": pd.to_datetime(d_str),
        "y": y_sales[:, store_idx],
        "is_weekend": data["W_t"],
        "discount": disc[:, store_idx],
    })

    # 2. 1-Hop Neighborhood signals
    if spatial_level in ("1hop", "2hop"):
        nbr_disc = disc @ P_spa[store_idx, :]
        y_lag1 = np.vstack([y_sales[0:1, :], y_sales[:-1, :]])
        nbr_ylag1 = (y_lag1 @ P_spa[store_idx, :]) / 100.0
        nbr_so = so_flag @ P_spa[store_idx, :]

        df_all["nbr_discount"] = nbr_disc
        df_all["nbr_ylag1"] = nbr_ylag1
        df_all["nbr_stockout"] = nbr_so

    # 3. 2-Hop Graph signals
    if spatial_level == "2hop":
        P_sq = P_spa @ P_spa
        nbr2_so = so_flag @ P_sq[store_idx, :]
        wh_state = data["node_w_state"][:, store_idx]

        df_all["nbr2_stockout"] = nbr2_so
        df_all["upstream_wh_so"] = wh_state

    # Unified holidays DataFrame
    h_dates = [dates[t].strftime("%Y-%m-%d") for t in range(T) if data["H_t"][t] == 1.0]
    holidays_df = pd.DataFrame({
        "holiday": "retail_holiday",
        "ds": pd.to_datetime(h_dates),
        "lower_window": 0,
        "upper_window": 0,
    })

    # Mask for uncensored training days (open and no stockout at store i)
    tr_mask = (t_arr < train_end) & (O_t == 1.0) & (so_flag[:, store_idx] == 0.0)
    df_train = df_all.iloc[tr_mask].copy()

    # Future test DataFrame (test period: train_end to T)
    te_mask = t_arr >= train_end
    df_future = df_all.iloc[te_mask].copy()

    return df_train, df_future, holidays_df


def fit_predict_prophet_spatial_ladder(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fits Prophet models across 10 stores for Base (0-Hop), 1-Hop, and 2-Hop specifications
    with native Fourier seasonality (weekly order=3, yearly order=5) and controlled regressors.
    """
    if not HAS_PROPHET:
        raise ImportError("Prophet is not installed. Please install prophet via pip or uv.")

    T = data["T"]
    N = data["N_stores"]
    train_end = data["train_end"]
    O_t = data["O_t"]

    results = {}

    # Parameters per store:
    # Base: 2 trend + 6 weekly + 10 yearly + 1 holiday + 2 regressors = 21 (all stores: 210)
    # 1Hop: 2 trend + 6 weekly + 10 yearly + 1 holiday + 5 regressors = 24 (all stores: 240)
    # 2Hop: 2 trend + 6 weekly + 10 yearly + 1 holiday + 7 regressors = 26 (all stores: 260)
    for level, spec_name, n_reg, n_par_per_store in [
        ("base", "Prophet_Base", 2, 21),
        ("1hop", "Prophet_1Hop", 5, 24),
        ("2hop", "Prophet_2Hop", 7, 26),
    ]:
        t0 = time.perf_counter()
        pred_demand = np.zeros((T, N))
        pred_sales = np.zeros((T, N))

        for i in range(N):
            df_tr, df_te, df_holidays = _build_prophet_store_dfs(data, store_idx=i, spatial_level=level)

            m = Prophet(
                growth="linear",
                holidays=df_holidays,
                weekly_seasonality=False,
                yearly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="additive",
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=1.0,
                holidays_prior_scale=1.0,
            )
            m.add_seasonality("weekly", period=7, fourier_order=3)
            m.add_seasonality("yearly", period=365.25, fourier_order=5)

            # Add all non-ds, non-y columns as regressors
            for col in df_tr.columns:
                if col not in ("ds", "y"):
                    m.add_regressor(col, prior_scale=1.0, standardize=False, mode="additive")

            m.fit(df_tr)
            fc = m.predict(df_te)

            pred_d_test = np.maximum(0.0, fc["yhat"].values)
            pred_demand[train_end:, i] = pred_d_test
            pred_sales[train_end:, i] = pred_d_test * O_t[train_end:]

        el_time = time.perf_counter() - t0
        results[spec_name] = {
            "pred_demand": pred_demand,
            "pred_sales": pred_sales,
            "runtime_sec": round(el_time, 4),
            "total_params": n_par_per_store * N,
            "active_dim": n_reg,
        }
        print(f"  [Prophet-Fourier] {spec_name}: 10 stores fit in {el_time:.2f}s (active_regressors={n_reg}, total_params={n_par_per_store*N})")

    return results


def fit_predict_prophet_public_dataset(
    ds_dict: Dict[str, Any],
    spatial_level: str = "base",
) -> Tuple[np.ndarray, float]:
    """
    Fit Prophet on a public benchmark dataset panel across its N stores with native Fourier weekly seasonality
    and controlled regressors (is_weekend, promo, cov1, cov2 + spatial neighborhood signals).
    """
    if not HAS_PROPHET:
        raise ImportError("Prophet is not installed.")

    T = ds_dict["T"]
    N = ds_dict["N"]
    train_end = ds_dict["train_end"]
    dates = ds_dict["dates"]
    y_obs = ds_dict["y_obs"]
    so_flag = ds_dict["stockout_flag"]
    promo = ds_dict["promo"]
    cov1 = ds_dict["cov1"]
    cov2 = ds_dict["cov2"]
    P_spa = ds_dict["P_spa"]
    stock_hours = ds_dict["stock_hours"]

    d_str = [d.strftime("%Y-%m-%d") for d in dates]
    y_lag1 = np.vstack([y_obs[0:1, :], y_obs[:-1, :]])

    t0 = time.perf_counter()
    pred_test = np.zeros((T - train_end, N))

    for i in range(N):
        df_all = pd.DataFrame({
            "ds": pd.to_datetime(d_str),
            "y": y_obs[:, i],
            "is_weekend": ds_dict["is_wknd"],
            "promo": promo[:, i],
            "cov1": cov1,
            "cov2": cov2,
        })

        if spatial_level == "1hop":
            nbr_p = promo @ P_spa[i, :]
            nbr_y = (y_lag1 @ P_spa[i, :]) / 100.0
            nbr_sh = (stock_hours @ P_spa[:, i]) / 16.0
            df_all["nbr_promo"] = nbr_p
            df_all["nbr_ylag1"] = nbr_y
            df_all["nbr_stock_hours"] = nbr_sh

        # Filter clean / in-stock training days
        tr_mask = (np.arange(T) < train_end) & (so_flag[:, i] == 0.0)
        df_tr = df_all.iloc[tr_mask].copy()

        te_mask = np.arange(T) >= train_end
        df_te = df_all.iloc[te_mask].copy()

        m = Prophet(
            growth="linear",
            weekly_seasonality=False,
            yearly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=1.0,
        )
        # Add native Fourier weekly seasonality
        m.add_seasonality("weekly", period=7, fourier_order=3)
        # For datasets covering at least 1 full year, add yearly seasonality
        if T >= 365:
            m.add_seasonality("yearly", period=365.25, fourier_order=5)

        for col in df_tr.columns:
            if col not in ("ds", "y"):
                m.add_regressor(col, prior_scale=1.0, standardize=False, mode="additive")

        m.fit(df_tr)
        fc = m.predict(df_te)
        pred_test[:, i] = np.maximum(0.0, fc["yhat"].values)

    runtime = time.perf_counter() - t0
    return pred_test, runtime
