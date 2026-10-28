"""
Linear, Pooled, Ridge, and Tobit Right-Censored MLE Models.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Any, Tuple


def fit_predict_linear_spec(
    spec_name: str,
    feat_dict: Dict[str, Any],
    data: Dict[str, Any],
    ridge_alpha: float = 0.0,
    stockout_flag_override: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Fit and predict a linear specification (local or pooled, all-open or uncensored).
    Returns: (pred_demand, pred_sales, runtime_sec, fitted_coef)
    """
    t0 = time.perf_counter()
    X = feat_dict["X"]
    mode = feat_dict["mode"]
    T, N, D = X.shape
    train_end = data["train_end"]
    O_t = data["O_t"]
    y_sales = data["y_sales"]
    V_it = data["V_it"]
    so_flag = stockout_flag_override if stockout_flag_override is not None else data["stockout_flag"]

    pred_demand = np.zeros((T, N))
    pred_sales = np.zeros((T, N))

    if mode.startswith("local"):
        coefs = []
        for i in range(N):
            Xi = X[:, i, :]
            tr_mask = (np.arange(T) < train_end) & (O_t == 1.0)
            if "uncensored" in mode:
                tr_mask = tr_mask & (so_flag[:, i] == 0.0)
            X_tr = Xi[tr_mask]
            y_tr = y_sales[tr_mask, i]
            if ridge_alpha > 0:
                reg = ridge_alpha * np.eye(D)
                reg[0, 0] = 0.0
                beta = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ y_tr)
            else:
                beta, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
            coefs.append(beta)
            pred_demand[:, i] = O_t * np.maximum(0.0, Xi @ beta)

            if "X_sales" in feat_dict:
                Xs_i = feat_dict["X_sales"][:, i, :]
                tr_open = (np.arange(T) < train_end) & (O_t == 1.0)
                beta_s, _, _, _ = np.linalg.lstsq(Xs_i[tr_open], y_sales[tr_open, i], rcond=None)
                pred_sales[:, i] = O_t * np.maximum(0.0, Xs_i @ beta_s)
            else:
                pred_sales[:, i] = O_t * np.minimum(pred_demand[:, i], V_it[:, i])
        coef_arr = np.array(coefs)
    else:
        tr_mask = (np.arange(T)[:, None] < train_end) & (O_t[:, None] == 1.0)
        if "uncensored" in mode:
            tr_mask = tr_mask & (so_flag == 0.0)
        X_tr = X[tr_mask]
        y_tr = y_sales[tr_mask]
        if ridge_alpha > 0:
            reg = ridge_alpha * np.eye(D)
            beta = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ y_tr)
        else:
            beta, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        coef_arr = beta
        for i in range(N):
            pred_demand[:, i] = O_t * np.maximum(0.0, X[:, i, :] @ beta)
            pred_sales[:, i] = O_t * np.minimum(pred_demand[:, i], V_it[:, i])

    elapsed = time.perf_counter() - t0
    return pred_demand, pred_sales, elapsed, coef_arr


def fit_predict_tobit_censored_mle(
    feat_dict: Dict[str, Any],
    data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a right-censored Tobit Normal Maximum Likelihood model per store on ALL open days.
    For uncensored observations (y_i,t < V_i,t): log phi((y - X beta)/sigma) - log(sigma)
    For right-censored observations (stockout == 1, y_i,t == V_i,t): log(1 - Phi((V - X beta)/sigma))
    Directly addresses Reviewer P1-8 (selection bias vs. censored likelihood).
    """
    t0 = time.perf_counter()
    X = feat_dict["X"]
    T, N, D = X.shape
    train_end = data["train_end"]
    O_t = data["O_t"]
    y_sales = data["y_sales"]
    V_it = data["V_it"]
    so_flag = data["stockout_flag"]

    pred_demand = np.zeros((T, N))
    pred_sales = np.zeros((T, N))

    for i in range(N):
        Xi = X[:, i, :]
        tr_open = (np.arange(T) < train_end) & (O_t == 1.0)
        X_tr = Xi[tr_open]
        y_tr = y_sales[tr_open, i]
        c_tr = (so_flag[tr_open, i] == 1.0)

        # Warm-start from uncensored OLS
        unc_mask = ~c_tr
        beta_init, _, _, _ = np.linalg.lstsq(X_tr[unc_mask], y_tr[unc_mask], rcond=None)
        resid = y_tr[unc_mask] - (X_tr[unc_mask] @ beta_init)
        log_sigma_init = np.log(max(2.0, float(np.std(resid))))
        params_init = np.append(beta_init, log_sigma_init)

        def neg_log_lik(params):
            beta = params[:-1]
            sigma = np.exp(np.clip(params[-1], -1.0, 4.5))
            mu = X_tr @ beta
            z = (y_tr - mu) / sigma
            # Uncensored log-likelihood
            ll_unc = -0.5 * (z[~c_tr] ** 2) - np.log(sigma)
            # Right-censored log survival function log(1 - Phi(z))
            ll_cen = norm.logsf(z[c_tr])
            return -(np.sum(ll_unc) + np.sum(ll_cen))

        res = minimize(neg_log_lik, params_init, method="L-BFGS-B", options={"maxiter": 200})
        beta_hat = res.x[:-1]
        pred_demand[:, i] = O_t * np.maximum(0.0, Xi @ beta_hat)
        pred_sales[:, i] = O_t * np.minimum(pred_demand[:, i], V_it[:, i])

    elapsed = time.perf_counter() - t0
    return pred_demand, pred_sales, elapsed
