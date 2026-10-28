"""
Two-Tier Evaluation Protocol:
1. Macro Level: Full Network Overall (n=270) + 3 Regimes (Normal, Cannibalization, Spillover)
2. Micro Level: Store-by-Store (S1..S10) x 4 Regimes (Overall_Store, Normal, Cannibalization, Spillover)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute WAPE (%), MAE, RMSE, and Bias (%) on masked entries."""
    yt = y_true[mask]
    yp = y_pred[mask]
    n_samples = int(np.sum(mask))
    if n_samples == 0 or np.sum(np.abs(yt)) < 1e-9:
        return {"n": 0, "wape": np.nan, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
    wape = float(np.sum(np.abs(yt - yp)) / np.sum(np.abs(yt)) * 100.0)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    bias = float(np.sum(yp - yt) / np.sum(yt) * 100.0)
    return {"n": n_samples, "wape": wape, "mae": mae, "rmse": rmse, "bias": bias}


def get_test_regime_masks(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return boolean masks (T, N_stores) for Overall Test and the 3 operational regimes."""
    T = data["T"]
    N = data["N_stores"]
    train_end = data["train_end"]
    O_t = data["O_t"]
    cannib = data["cannibalization"]
    spill = data["spillover"]
    V_it = data["V_it"]

    test_open = (np.arange(T)[:, None] >= train_end) & (O_t[:, None] == 1.0)
    test_open = np.broadcast_to(test_open, (T, N))

    mask_cannib = test_open & (cannib > 5.0)
    mask_spill = test_open & (spill > 5.0)
    mask_shock = test_open & ((cannib > 5.0) | (spill > 5.0) | (V_it < 100.0))
    mask_normal = test_open & (~mask_shock)

    return {
        "Overall": test_open,
        "Normal": mask_normal,
        "Cannibalization": mask_cannib,
        "Spillover": mask_spill,
    }


def evaluate_model_macro_and_micro(
    model_id: str,
    pred_demand: np.ndarray,
    data: Dict[str, Any],
    pred_sales: np.ndarray = None,
    meta: Dict[str, Any] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate a model at both Macro (Network + Regimes) and Micro (Per-Store x Regimes) levels.
    """
    if pred_sales is None:
        pred_sales = data["O_t"][:, None] * np.minimum(pred_demand, data["V_it"])

    masks = get_test_regime_masks(data)
    D_sim = data["D_sim"]
    mu_open = data["mu_open"]
    y_sales = data["y_sales"]

    # 1. Macro summary row
    macro_row = {"model_id": model_id}
    if meta:
        macro_row.update(meta)

    for r_name, r_mask in masks.items():
        m_d = compute_metrics(D_sim, pred_demand, r_mask)
        macro_row[f"{r_name}_n"] = m_d["n"]
        macro_row[f"{r_name}_D_wape"] = round(m_d["wape"], 3)
        macro_row[f"{r_name}_D_mae"] = round(m_d["mae"], 3)
        macro_row[f"{r_name}_D_bias"] = round(m_d["bias"], 3)

    m_mu = compute_metrics(mu_open, pred_demand, masks["Overall"])
    m_ys = compute_metrics(y_sales, pred_sales, masks["Overall"])
    macro_row["Overall_Mu_wape"] = round(m_mu["wape"], 3)
    macro_row["Overall_Mu_mae"] = round(m_mu["mae"], 3)
    macro_row["Overall_Mu_bias"] = round(m_mu["bias"], 3)
    macro_row["Overall_Sales_wape"] = round(m_ys["wape"], 3)
    macro_row["Overall_Sales_mae"] = round(m_ys["mae"], 3)
    macro_row["Overall_Sales_bias"] = round(m_ys["bias"], 3)

    # 2. Micro Store-by-Store x Regime rows
    micro_rows = []
    N = data["N_stores"]
    for i in range(N):
        s_id = data["store_ids"][i]
        c_name = data["cluster_names"][i]
        role = data["store_roles"][i]
        w_id = f"W{data['w_assign'][i] + 1}"

        for r_name, r_mask in masks.items():
            s_mask = np.zeros_like(r_mask, dtype=bool)
            s_mask[:, i] = r_mask[:, i]
            m_sd = compute_metrics(D_sim, pred_demand, s_mask)
            m_smu = compute_metrics(mu_open, pred_demand, s_mask)
            m_sys = compute_metrics(y_sales, pred_sales, s_mask)

            micro_rows.append({
                "model_id": model_id,
                "store_id": s_id,
                "cluster": c_name,
                "warehouse": w_id,
                "role": role,
                "regime": r_name,
                "n_days": m_sd["n"],
                "D_wape": round(m_sd["wape"], 3) if not np.isnan(m_sd["wape"]) else np.nan,
                "D_mae": round(m_sd["mae"], 3) if not np.isnan(m_sd["mae"]) else np.nan,
                "D_bias": round(m_sd["bias"], 3) if not np.isnan(m_sd["bias"]) else np.nan,
                "Mu_wape": round(m_smu["wape"], 3) if not np.isnan(m_smu["wape"]) else np.nan,
                "Sales_wape": round(m_sys["wape"], 3) if not np.isnan(m_sys["wape"]) else np.nan,
            })

    return macro_row, micro_rows
