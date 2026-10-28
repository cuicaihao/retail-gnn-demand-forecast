"""
Gradient Boosted Decision Trees (GBDT) Module.
Uses native LightGBM if installed, and includes a fast, exact vectorized NumPy
Histogram CART Gradient Boosting implementation for zero-dependency reproducibility.
"""
import time
import numpy as np
from typing import Dict, Any, Tuple, List

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class _FastCARTNode:
    __slots__ = ("is_leaf", "value", "feat_idx", "thresh", "left", "right")

    def __init__(self, is_leaf=True, value=0.0, feat_idx=-1, thresh=0.0, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feat_idx = feat_idx
        self.thresh = thresh
        self.left = left
        self.right = right


class NumPyHistGBDTRegressor:
    """
    Vectorized Histogram-based Gradient Boosted Decision Tree Regressor in pure NumPy.
    Supports max_depth, min_samples_leaf, subsample, colsample_bytree, and quantile binning.
    """

    def __init__(
        self,
        n_estimators: int = 90,
        learning_rate: float = 0.09,
        max_depth: int = 5,
        min_samples_leaf: int = 18,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        n_bins: int = 16,
        l2_reg: float = 1.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_bins = n_bins
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.base_value = 0.0
        self.trees: List[_FastCARTNode] = []
        self.feature_importances_: np.ndarray = None

    def _build_tree(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        idx: np.ndarray,
        depth: int,
        feat_candidates: np.ndarray,
        thresholds_list: List[np.ndarray],
    ) -> _FastCARTNode:
        n_node = len(idx)
        leaf_val = float(np.sum(grad[idx]) / (n_node + self.l2_reg))
        if depth >= self.max_depth or n_node < 2 * self.min_samples_leaf:
            return _FastCARTNode(is_leaf=True, value=leaf_val)

        g_node = grad[idx]
        sum_g = float(np.sum(g_node))
        parent_gain = (sum_g * sum_g) / (n_node + self.l2_reg)

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0

        X_node = X[idx]
        for f in feat_candidates:
            ths = thresholds_list[f]
            if len(ths) == 0:
                continue
            col = X_node[:, f]
            # Vectorized comparison over candidate thresholds: col[:, None] <= ths[None, :]
            mask_mat = col[:, None] <= ths[None, :]
            n_left = mask_mat.sum(axis=0)
            n_right = n_node - n_left
            valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not np.any(valid):
                continue
            g_left = (g_node[:, None] * mask_mat).sum(axis=0)
            g_right = sum_g - g_left
            gains = (
                (g_left[valid] ** 2) / (n_left[valid] + self.l2_reg)
                + (g_right[valid] ** 2) / (n_right[valid] + self.l2_reg)
                - parent_gain
            )
            max_idx = int(np.argmax(gains))
            if gains[max_idx] > best_gain:
                best_gain = float(gains[max_idx])
                best_feat = int(f)
                best_thresh = float(ths[valid][max_idx])

        if best_feat == -1 or best_gain <= 1e-6:
            return _FastCARTNode(is_leaf=True, value=leaf_val)

        self.feature_importances_[best_feat] += best_gain
        left_mask = X_node[:, best_feat] <= best_thresh
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]

        left_child = self._build_tree(X, grad, left_idx, depth + 1, feat_candidates, thresholds_list)
        right_child = self._build_tree(X, grad, right_idx, depth + 1, feat_candidates, thresholds_list)
        return _FastCARTNode(
            is_leaf=False, value=leaf_val, feat_idx=best_feat, thresh=best_thresh,
            left=left_child, right=right_child
        )

    def _predict_tree(self, node: _FastCARTNode, X: np.ndarray, idx: np.ndarray, out: np.ndarray):
        if node.is_leaf or len(idx) == 0:
            out[idx] = node.value
            return
        mask = X[idx, node.feat_idx] <= node.thresh
        self._predict_tree(node.left, X, idx[mask], out)
        self._predict_tree(node.right, X, idx[~mask], out)

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        N, D = X.shape
        self.base_value = float(np.mean(y))
        y_hat = np.full(N, self.base_value)
        self.feature_importances_ = np.zeros(D)

        # Precompute candidate split thresholds per feature
        thresholds_list = []
        for f in range(D):
            uniq = np.unique(X[:, f])
            if len(uniq) <= 1:
                thresholds_list.append(np.array([]))
            elif len(uniq) <= self.n_bins:
                thresholds_list.append(0.5 * (uniq[:-1] + uniq[1:]))
            else:
                qs = np.linspace(0.03, 0.97, self.n_bins)
                thresholds_list.append(np.unique(np.quantile(X[:, f], qs)))

        all_indices = np.arange(N)
        n_sub = max(2 * self.min_samples_leaf, int(N * self.subsample))
        n_col = max(1, int(np.ceil(D * self.colsample_bytree)))

        self.trees = []
        for _ in range(self.n_estimators):
            residual = y - y_hat
            sub_idx = rng.choice(all_indices, size=n_sub, replace=False) if n_sub < N else all_indices
            feat_cand = rng.choice(D, size=n_col, replace=False) if n_col < D else np.arange(D)
            tree = self._build_tree(X, residual, sub_idx, 0, feat_cand, thresholds_list)
            self.trees.append(tree)
            step_pred = np.zeros(N)
            self._predict_tree(tree, X, all_indices, step_pred)
            y_hat += self.learning_rate * step_pred

        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        y_hat = np.full(N, self.base_value)
        all_idx = np.arange(N)
        step_pred = np.zeros(N)
        for tree in self.trees:
            self._predict_tree(tree, X, all_idx, step_pred)
            y_hat += self.learning_rate * step_pred
        return y_hat


def fit_predict_gbdt_multi_seed(
    feat_dict: Dict[str, Any],
    data: Dict[str, Any],
    seeds: List[int] = [42, 101, 202, 303, 404],
    hyperparams: Dict[str, Any] = None,
    engine: str = "lgbm",
) -> Dict[str, Any]:
    """
    Fit GBDT across multiple random seeds on uncensored training data.
    Supports engine="lgbm", engine="xgboost", or engine="numpy".
    Returns mean prediction, per-seed predictions, feature importances, and runtime.
    """
    if hyperparams is None:
        hyperparams = {
            "n_estimators": 90,
            "learning_rate": 0.09,
            "max_depth": 5,
            "min_samples_leaf": 18,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        }

    X = feat_dict["X"]
    T, N, D = X.shape
    train_end = data["train_end"]
    O_t = data["O_t"]
    y_sales = data["y_sales"]
    V_it = data["V_it"]
    so_flag = data["stockout_flag"]

    tr_mask = (np.arange(T)[:, None] < train_end) & (O_t[:, None] == 1.0) & (so_flag == 0.0)
    X_tr = X[tr_mask]
    y_tr = y_sales[tr_mask]
    X_all_flat = X.reshape(T * N, D)

    seed_preds = []
    importances = []
    runtimes = []

    actual_engine = engine
    if engine == "xgboost" and not HAS_XGB:
        actual_engine = "lgbm" if HAS_LGBM else "numpy"
    elif engine == "lgbm" and not HAS_LGBM:
        actual_engine = "xgboost" if HAS_XGB else "numpy"

    for sd in seeds:
        t0 = time.perf_counter()
        if actual_engine == "xgboost" and HAS_XGB:
            model = xgb.XGBRegressor(
                n_estimators=hyperparams["n_estimators"],
                learning_rate=hyperparams["learning_rate"],
                max_depth=hyperparams["max_depth"],
                subsample=hyperparams["subsample"],
                colsample_bytree=hyperparams["colsample_bytree"],
                tree_method="hist",
                random_state=sd,
                verbosity=0,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            pred_flat = model.predict(X_all_flat)
            imp = model.feature_importances_ / max(1e-9, np.sum(model.feature_importances_))
        elif actual_engine == "lgbm" and HAS_LGBM:
            model = lgb.LGBMRegressor(
                n_estimators=hyperparams["n_estimators"],
                learning_rate=hyperparams["learning_rate"],
                max_depth=hyperparams["max_depth"],
                num_leaves=min(31, 2 ** hyperparams["max_depth"] - 1),
                min_child_samples=hyperparams["min_samples_leaf"],
                subsample=hyperparams["subsample"],
                colsample_bytree=hyperparams["colsample_bytree"],
                random_state=sd,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            pred_flat = model.predict(X_all_flat)
            imp = model.feature_importances_ / max(1e-9, np.sum(model.feature_importances_))
        else:
            model = NumPyHistGBDTRegressor(
                n_estimators=hyperparams["n_estimators"],
                learning_rate=hyperparams["learning_rate"],
                max_depth=hyperparams["max_depth"],
                min_samples_leaf=hyperparams["min_samples_leaf"],
                subsample=hyperparams["subsample"],
                colsample_bytree=hyperparams["colsample_bytree"],
                random_state=sd,
            )
            model.fit(X_tr, y_tr)
            pred_flat = model.predict(X_all_flat)
            imp = model.feature_importances_

        el = time.perf_counter() - t0
        runtimes.append(el)
        pred_d = O_t[:, None] * np.maximum(0.0, pred_flat.reshape(T, N))
        seed_preds.append(pred_d)
        importances.append(imp)

    seed_preds_arr = np.stack(seed_preds, axis=0)
    mean_pred_demand = np.mean(seed_preds_arr, axis=0)
    mean_pred_sales = O_t[:, None] * np.minimum(mean_pred_demand, V_it)
    mean_imp = np.mean(np.stack(importances, axis=0), axis=0)

    engine_display = "XGBoost" if actual_engine == "xgboost" else ("LightGBM" if actual_engine == "lgbm" else "NumPyHistGBDT")

    return {
        "mean_pred_demand": mean_pred_demand,
        "mean_pred_sales": mean_pred_sales,
        "seed_preds": seed_preds_arr,
        "feature_importances": mean_imp,
        "mean_runtime_sec": float(np.mean(runtimes)),
        "engine": engine_display,
    }
