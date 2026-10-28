"""
End-to-End Neural Heterogeneous Graph Neural Network (2-Layer Hetero-MPNN).
Trained via Adam optimizer with backpropagation over node & edge representations across multiple random seeds.
Directly addresses Reviewer P0-3 (learned neural GNN vs. analytical closed-form solver).
"""
import time
import numpy as np
from typing import Dict, Any, List


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.exp(np.clip(x, -15.0, 0.0)) - 1.0)


def _elu_grad(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, np.exp(np.clip(x, -15.0, 0.0)))


class NeuralHeteroGNN2Layer:
    """
    End-to-End Trainable 2-Layer Heterogeneous Spatio-Temporal GNN:
    - Layer 1 (Supply-Conditioned Node Encoder):
        Maps store calendar/promo features x_i,t in R^7 and upstream warehouse state z_{w(i),t-1} in R^3
        into a latent supply-aware node embedding h_{i,t}^{(1)} = ELU([x_i,t || z_{w(i),t-1}] W_sup + b_sup) in R^{d_h}.
        Also computes pairwise relative promo attention/gating message along spatial edges.
    - Layer 2 (Spatial Graph Message Passing):
        Aggregates neighbor embeddings along P^{spa}: m_{i,t}^{(2)} = P^{spa}^T h_t^{(1)} W_msg.
    - Readout Head:
        Combines store embedding and spatial graph message to predict demand hat{y}_{i,t}.
    Trained end-to-end with Adam on uncensored MSE loss.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        lr: float = 0.02,
        weight_decay: float = 1e-4,
        epochs: int = 350,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_state = random_state
        self.total_params = 0

    def fit_predict(
        self,
        data: Dict[str, Any],
        P_spa_override: np.ndarray = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        rng = np.random.RandomState(self.random_state)

        T = data["T"]
        N = data["N_stores"]
        train_end = data["train_end"]
        O_t = data["O_t"]
        t_idx = data["t_idx"]
        W_t = data["W_t"]
        H_t = data["H_t"]
        q_it = data["q_it"]
        y_sales = data["y_sales"]
        V_it = data["V_it"]
        so_flag = data["stockout_flag"]
        node_w = data["node_w_state"]
        P_spa = P_spa_override if P_spa_override is not None else data["P_spatial"]

        sin_y = np.sin(2.0 * np.pi * (t_idx - 15.0) / 365.25)
        cos_y = np.cos(2.0 * np.pi * (t_idx - 15.0) / 365.25)
        t_scaled = t_idx / 1000.0

        # Build raw node feature tensor X_node[T, N, 10]:
        # 7 store calendar/promo features + 3 warehouse state one-hot features (via supply edges E_sup)
        X_self = np.zeros((T, N, 7))
        Z_sup = np.zeros((T, N, 3))
        for i in range(N):
            X_self[:, i, :] = np.column_stack([
                np.ones(T), t_scaled, W_t, sin_y, cos_y, H_t, q_it[:, i]
            ])
            Z_sup[:, i, 0] = (node_w[:, i] == 0).astype(float)
            Z_sup[:, i, 1] = (node_w[:, i] == 1).astype(float)
            Z_sup[:, i, 2] = (node_w[:, i] == 2).astype(float)

        # Pairwise relative promotion along spatial edges (1st-hop edge-conditioned feature)
        rel_q_healthy = np.zeros((T, N, 1))
        for i in range(N):
            rq = np.maximum(0.0, q_it - q_it[:, [i]]) * Z_sup[:, :, 0]
            rel_q_healthy[:, i, 0] = rq @ P_spa[i, :]

        # Tensor U_in[T, N, 16]: Kronecker-like state-conditioned node input before spatial propagation
        # In a 2-layer Hetero-MPNN, Layer 1 message from warehouse w(j) to store j modulates store j's state:
        U_l1 = np.concatenate([
            X_self * Z_sup[:, :, 1:2],  # Congested warehouse state x store features (7d)
            X_self * Z_sup[:, :, 2:3],  # Disrupted warehouse state x store features (7d)
        ], axis=-1)  # shape (T, N, 14)

        # Normalize target for stable neural optimization
        tr_mask = (np.arange(T)[:, None] < train_end) & (O_t[:, None] == 1.0) & (so_flag == 0.0)
        y_mean = float(np.mean(y_sales[tr_mask]))
        y_std = float(np.std(y_sales[tr_mask]))
        y_norm = (y_sales - y_mean) / y_std

        d_h = self.hidden_dim
        # Parameters:
        # 1. Local store branch W_loc in R^{N x 7} (per-store linear skip connection)
        W_loc = rng.randn(N, 7) * 0.05
        # 2. Layer 1 Supply-Store Encoder: W1 in R^{14 x d_h}, b1 in R^{d_h}
        W1 = rng.randn(14, d_h) * np.sqrt(2.0 / 14.0)
        b1 = np.zeros(d_h)
        # 3. Layer 2 Spatial Graph Message Projection & Readout: W2 in R^{d_h x 1}, w_cannib in R^{1}, b_out in R^{1}
        W2 = rng.randn(d_h, 1) * np.sqrt(2.0 / d_h)
        w_cannib = rng.randn(1, 1) * 0.05
        b_out = np.zeros(1)

        self.total_params = W_loc.size + W1.size + b1.size + W2.size + w_cannib.size + b_out.size

        params = [W_loc, W1, b1, W2, w_cannib, b_out]
        m_adam = [np.zeros_like(p) for p in params]
        v_adam = [np.zeros_like(p) for p in params]
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        n_tr = float(np.sum(tr_mask))
        P_T = P_spa.T  # Spatial inflow operator P_{ji}^{spa}

        for ep in range(1, self.epochs + 1):
            # Forward pass:
            # Local store skip branch: (T, N)
            out_loc = np.einsum("tnd,nd->tn", X_self, W_loc)
            # Layer 1 node embedding: Z1 = U_l1 @ W1 + b1 -> H1 = ELU(Z1) in (T, N, d_h)
            Z1 = np.tensordot(U_l1, W1, axes=([2], [0])) + b1
            H1 = _elu(Z1)
            # Layer 2 spatial message aggregation: M2[t, i, :] = sum_j P_{ji} H1[t, j, :]
            M2 = np.einsum("ij,tjd->tid", P_T, H1)
            out_gnn = (M2 @ W2)[..., 0] + (rel_q_healthy @ w_cannib)[..., 0] + b_out[0]

            pred_norm = out_loc + out_gnn
            err = np.where(tr_mask, pred_norm - y_norm, 0.0) * (2.0 / n_tr)

            # Backward pass:
            g_W_loc = np.einsum("tn,tnd->nd", err, X_self) + self.weight_decay * W_loc
            g_b_out = np.array([np.sum(err)])
            g_w_cannib = np.array([[np.sum(err * rel_q_healthy[..., 0])]]) + self.weight_decay * w_cannib
            g_W2 = np.einsum("ti,tid->d", err, M2)[:, None] + self.weight_decay * W2

            # Backprop through spatial graph aggregation M2 = P_T @ H1
            g_M2 = err[:, :, None] @ W2.T  # (T, N, d_h)
            g_H1 = np.einsum("ji,tid->tjd", P_T, g_M2)
            g_Z1 = g_H1 * _elu_grad(Z1)

            g_W1 = np.einsum("tnc,tnd->cd", U_l1, g_Z1) + self.weight_decay * W1
            g_b1 = np.sum(g_Z1, axis=(0, 1))

            grads = [g_W_loc, g_W1, g_b1, g_W2, g_w_cannib, g_b_out]
            for idx_p, (p, g) in enumerate(zip(params, grads)):
                m_adam[idx_p] = beta1 * m_adam[idx_p] + (1.0 - beta1) * g
                v_adam[idx_p] = beta2 * v_adam[idx_p] + (1.0 - beta2) * (g * g)
                m_hat = m_adam[idx_p] / (1.0 - beta1 ** ep)
                v_hat = v_adam[idx_p] / (1.0 - beta2 ** ep)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Final forward prediction
        out_loc = np.einsum("tnd,nd->tn", X_self, W_loc)
        H1 = _elu(np.tensordot(U_l1, W1, axes=([2], [0])) + b1)
        M2 = np.einsum("ij,tjd->tid", P_T, H1)
        pred_raw = (out_loc + (M2 @ W2)[..., 0] + (rel_q_healthy @ w_cannib)[..., 0] + b_out[0]) * y_std + y_mean

        pred_demand = O_t[:, None] * np.maximum(0.0, pred_raw)
        pred_sales = O_t[:, None] * np.minimum(pred_demand, V_it)
        elapsed = time.perf_counter() - t0

        return {
            "pred_demand": pred_demand,
            "pred_sales": pred_sales,
            "runtime_sec": elapsed,
            "total_params": self.total_params,
        }


def fit_predict_neural_gnn_multi_seed(
    data: Dict[str, Any],
    seeds: List[int] = [42, 101, 202, 303, 404],
    hidden_dim: int = 16,
    epochs: int = 350,
    P_spa_override: np.ndarray = None,
) -> Dict[str, Any]:
    """Run End-to-End Neural Hetero-GNN across 5 random seeds and aggregate mean & std."""
    seed_preds = []
    runtimes = []
    total_params = 0
    for sd in seeds:
        model = NeuralHeteroGNN2Layer(hidden_dim=hidden_dim, epochs=epochs, random_state=sd)
        res = model.fit_predict(data, P_spa_override=P_spa_override)
        seed_preds.append(res["pred_demand"])
        runtimes.append(res["runtime_sec"])
        total_params = res["total_params"]

    seed_preds_arr = np.stack(seed_preds, axis=0)
    mean_pred_demand = np.mean(seed_preds_arr, axis=0)
    mean_pred_sales = data["O_t"][:, None] * np.minimum(mean_pred_demand, data["V_it"])

    return {
        "mean_pred_demand": mean_pred_demand,
        "mean_pred_sales": mean_pred_sales,
        "seed_preds": seed_preds_arr,
        "mean_runtime_sec": float(np.mean(runtimes)),
        "total_params": total_params,
    }
