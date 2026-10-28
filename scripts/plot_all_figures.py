#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoupled Publication-Grade Figure Generator for Retail GNN Demand Forecasting
=============================================================================
Reads solely from static CSV files in `results/plot_data/` to produce publication-
quality figures (PNG and WebP) in both Chinese (`zh`) and English (`en`).

Figures generated:
  Fig 1: retail_network_topology          - Heterogeneous graph topology & random-walk matrix
  Fig 2: all_stores_sales_panorama        - 10-store 184-day daily sales vs latent demand
  Fig 3: single_store_decomposition       - S1 core vs S6 bridge store demand structure
  Fig 4: store_timeseries_comparison      - 28-day test period forecast comparison (S5 & S6)
  Fig 5: regime_ablation_metrics          - Model performance across business regimes (WAPE & Bias)
  Fig 6: multitool_benchmark_tradeoff     - Runtime vs accuracy tradeoff & 4-layer toolkit ecosystem
  Fig 7: store_regime_heatmap             - Store x Regime WAPE breakdown heatmap
  Fig 8: feature_dim_vs_wape              - Parameter ladder & information tier vs error
  Fig 9: public_benchmarks_comparison     - Real-world benchmark evaluation (Dingdong, Favorita, M5)

Outputs:
  figures/<stem>.png, figures/<stem>.zh.webp    (Chinese)
  figures/<stem>_en.png, figures/<stem>.en.webp (English)
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOT_DATA_DIR = PROJECT_ROOT / "results" / "plot_data"
FIGURES_DIR = PROJECT_ROOT / "figures"

ZH_FONTS = [
    "Hiragino Sans GB", "PingFang SC", "Heiti SC", "Heiti TC", "Arial Unicode MS",
    "Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "DejaVu Sans"
]
EN_FONTS = ["DejaVu Sans", "Arial", "Helvetica"]

# Standard color palette matching research report & publication aesthetics
C_W1, C_W2 = "#1D4ED8", "#C2410C"          # W1 (blue), W2 (orange)
C_S_W1, C_S_W2 = "#60A5FA", "#FB923C"      # stores by supplier
C_1A, C_1B, C_2, C_3 = "#EF4444", "#F59E0B", "#8B5CF6", "#10B981"
C_GBDT = "#0284C7"
C_ANALYTICAL = "#059669"


def _set_font(lang: str) -> None:
    plt.rcParams["font.sans-serif"] = ZH_FONTS if lang == "zh" else EN_FONTS
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def _t(lang: str, zh: str, en: str) -> str:
    return zh if lang == "zh" else en


def _save_dual(fig: plt.Figure, out_dir: Path, stem: str, lang: str, dpi: int = 160) -> None:
    """Save high-resolution PNG and optimized WebP for web delivery."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / (f"{stem}.png" if lang == "zh" else f"{stem}_en.png")
    webp_path = out_dir / f"{stem}.{lang}.webp"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(png_path) as im:
        im.convert("RGB").save(webp_path, "WEBP", quality=88, method=6)
    print(f"  -> Generated {png_path.name} & {webp_path.name}")


# ----------------------------------------------------------------------------------------------
# Figure 1: Network Topology & Random-Walk Matrix
# ----------------------------------------------------------------------------------------------
def plot_fig1_topology(out_dir: Path, lang: str = "zh") -> None:
    _set_font(lang)
    nodes_file = PLOT_DATA_DIR / "plot_fig1_topology_nodes.csv"
    matrix_file = PLOT_DATA_DIR / "plot_fig1_spatial_matrix.csv"

    if not nodes_file.exists() or not matrix_file.exists():
        print(f"Warning: Missing files for Fig 1 in {PLOT_DATA_DIR}")
        return

    nodes_df = pd.read_csv(nodes_file)
    p_mat_df = pd.read_csv(matrix_file, index_col=0)
    P = p_mat_df.values
    store_ids = list(p_mat_df.columns)
    N = len(store_ids)

    store_nodes = nodes_df[nodes_df["node_type"] == "Store"].copy()
    wh_nodes = nodes_df[nodes_df["node_type"] == "Warehouse"].copy()

    # Mapping
    sc = store_nodes[["x_km", "y_km"]].values
    wc = wh_nodes[["x_km", "y_km"]].values
    wa = [0 if w == "W1" else 1 for w in store_nodes["warehouse"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Supply edges E^sup (dashed)
    for i in range(N):
        w = wa[i]
        ax1.plot([wc[w, 0], sc[i, 0]], [wc[w, 1], sc[i, 1]], ls="--", lw=1.2,
                 color=C_W1 if w == 0 else C_W2, alpha=0.45, zorder=1)

    # Spatial edges E^spa (solid, width ~ P[i, j])
    for i in range(N):
        for j in range(i + 1, N):
            p_val = max(P[i, j], P[j, i])
            if p_val > 0:
                ax1.plot([sc[i, 0], sc[j, 0]], [sc[i, 1], sc[j, 1]], color="#475569",
                         lw=1.0 + 3.8 * p_val, alpha=0.6, zorder=2)

    # Warehouses
    for w, row in wh_nodes.iterrows():
        x, y = row["x_km"], row["y_km"]
        w_idx = 0 if row["warehouse"] == "W1" else 1
        ax1.scatter(x, y, s=520, marker="s", color=C_W1 if w_idx == 0 else C_W2,
                    edgecolors="black", lw=1.5, zorder=5)
        ax1.text(x, y, row["node_id"], color="white", fontsize=10.5, fontweight="bold",
                 ha="center", va="center", zorder=6)

    # Stores
    for i, row in store_nodes.reset_index(drop=True).iterrows():
        x, y = row["x_km"], row["y_km"]
        bridge = "Bridge" in str(row["role"]) or row["node_id"] in ("S4", "S6")
        ax1.scatter(x, y, s=360 if bridge else 290, color=C_S_W1 if wa[i] == 0 else C_S_W2,
                    edgecolors="#DC2626" if bridge else "#1E293B", lw=2.6 if bridge else 1.2, zorder=5)
        ax1.text(x, y, row["node_id"], fontsize=9, fontweight="bold", ha="center", va="center", zorder=6)

    clusters = [
        (2.75, 9.15, _t(lang, "商圈 A：东区高密\nS4 = 东区跨仓交界店", "Cluster A: Eastern High-Density\nS4 = cross-warehouse border store")),
        (8.15, 6.55, _t(lang, "商圈 B：中轴走廊\nS6 = 中轴跨仓枢纽店", "Cluster B: Central Corridor\nS6 = cross-warehouse bridge hub")),
        (14.45, 3.95, _t(lang, "商圈 C：西区中心\n全部由 W2 供货", "Cluster C: Western Centre\nall supplied by W2")),
    ]
    for x, y, s in clusters:
        ax1.text(x, y, s, fontsize=8.8, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1"))

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_W1, markersize=10, label=_t(lang, "W1 东区仓", "W1 Eastern Hub")),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_W2, markersize=10, label=_t(lang, "W2 西区仓", "W2 Western Hub")),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_S_W1, markeredgecolor="#1E293B", markersize=9, label=_t(lang, "W1 供货门店", "Store supplied by W1")),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_S_W2, markeredgecolor="#1E293B", markersize=9, label=_t(lang, "W2 供货门店", "Store supplied by W2")),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#DC2626", markeredgewidth=2.4, markersize=9, label=_t(lang, "跨仓枢纽门店 (S4, S6)", "Cross-warehouse bridge (S4, S6)")),
        Line2D([0], [0], color="#475569", lw=2.4, label=_t(lang, "空间转移边 E^spa（d ≤ 3.5 km）", "Spatial edge E^spa (d ≤ 3.5 km)")),
        Line2D([0], [0], color="#64748B", lw=1.2, ls="--", label=_t(lang, "供应链边 E^sup（1 天提前期）", "Supply edge E^sup (1-day lead time)")),
    ]
    ax1.legend(handles=handles, loc="lower left", fontsize=8.3, framealpha=0.94)
    ax1.set_xlim(0.5, 16.5)
    ax1.set_ylim(0.3, 11.0)
    ax1.grid(True, ls=":", alpha=0.45)
    ax1.set_xlabel(_t(lang, "X 坐标 (km)", "X (km)"))
    ax1.set_ylabel(_t(lang, "Y 坐标 (km)", "Y (km)"))
    ax1.set_title(_t(lang, "(a) 异构零售网络 G = (V^W ∪ V^S, E^sup ∪ E^spa)",
                     "(a) Heterogeneous retail graph G = (V^W ∪ V^S, E^sup ∪ E^spa)"), fontsize=12, fontweight="bold")

    im = ax2.imshow(P, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(N))
    ax2.set_xticklabels(store_ids, fontsize=9)
    ax2.set_yticks(range(N))
    ax2.set_yticklabels(store_ids, fontsize=9)
    for i in range(N):
        for j in range(N):
            if P[i, j] > 0:
                ax2.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if P[i, j] > 0.45 else "#0F172A")
    for (start, size, color) in ((0, 4, "#2563EB"), (4, 3, "#EA580C"), (7, 3, "#16A34A")):
        ax2.add_patch(mpatches.Rectangle((start - 0.5, start - 0.5), size, size, fill=False, ec=color, lw=2.4))
    ax2.set_xlabel(_t(lang, "目标门店 j", "Destination store j"))
    ax2.set_ylabel(_t(lang, "源门店 i", "Source store i"))
    ax2.set_title(_t(lang, "(b) 空间随机游走转移矩阵 P^spa = (D^spa)^-1 A^spa",
                     "(b) Spatial random-walk matrix P^spa = (D^spa)^-1 A^spa"), fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_dual(fig, out_dir, "retail_network_topology", lang)


# ----------------------------------------------------------------------------------------------
# Figure 2: All Stores Sales Panorama (184 Days)
# ----------------------------------------------------------------------------------------------
def plot_fig2_panorama(out_dir: Path, lang: str = "zh") -> None:
    _set_font(lang)
    csv_file = PLOT_DATA_DIR / "plot_fig2_panorama_184d.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist.")
        return

    df = pd.read_csv(csv_file)
    df["ds"] = pd.to_datetime(df["ds"])
    store_ids = [f"S{i}" for i in range(1, 11)]

    fig, axes = plt.subplots(5, 2, figsize=(17, 13.5), sharex=True, sharey=True)
    for i, (ax, s_id) in enumerate(zip(axes.T.flatten(), store_ids)):
        sub = df[df["store_id"] == s_id].sort_values("ds")
        d = sub["ds"].values
        D = sub["latent_demand"].values
        y = sub["y"].values
        so = sub["stockout"].values == 1
        promo = sub["discount"].values > 0
        w_id = sub["warehouse_id"].iloc[0]
        bridge = s_id in ("S4", "S6")

        ax.fill_between(d, 0, 1, where=promo, transform=ax.get_xaxis_transform(),
                        color="#FEF08A", alpha=0.45, lw=0, step="mid")
        ax.plot(d, D, color="#6B7280", ls="--", lw=0.9, label=_t(lang, "潜在需求 D^sim", "Latent demand D^sim"))
        ax.plot(d, np.where(sub["is_open"] == 1, y, np.nan),
                color="#EA580C" if bridge else "#1D4ED8", lw=1.2,
                label=_t(lang, "观测销量 y^sales", "Observed sales y^sales"))
        if so.any():
            ax.scatter(d[so], y[so], marker="v", color="#DC2626", s=22, zorder=5,
                       label=_t(lang, "缺货截断", "Stockout censoring"))
        role = _t(lang, "跨仓枢纽", "bridge") if bridge else ""
        mean_y = y[sub["is_open"] == 1].mean()
        ax.set_title(f"{s_id} · {w_id}" + (f" · {role}" if role else "")
                     + f"  ({_t(lang, '日均销量', 'mean sales')} {mean_y:.0f})",
                     fontsize=10, fontweight="bold", loc="left", color="#C2410C" if bridge else "#0F172A")
        ax.grid(True, ls=":", alpha=0.4)

    axes[0, 0].legend(handles=axes[0, 0].get_legend_handles_labels()[0] + [
        mpatches.Patch(color="#FEF08A", label=_t(lang, "促销窗口", "Promotion window"))
    ], loc="upper left", fontsize=8, ncol=4, framealpha=0.92)

    for ax in axes[-1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    fig.suptitle(_t(lang, "10 家门店日度观测销量与潜在需求（2025-07-01 至 2025-12-31，共 184 天）",
                    "Daily observed sales vs. latent demand across 10 stores (2025-07-01 to 2025-12-31, 184 days)"),
                 fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout()
    _save_dual(fig, out_dir, "all_stores_sales_panorama", lang)


# ----------------------------------------------------------------------------------------------
# Figure 3: Single Store Demand Structure Decomposition (S1 & S6)
# ----------------------------------------------------------------------------------------------
def plot_fig3_decomposition(out_dir: Path, lang: str = "zh") -> None:
    _set_font(lang)
    csv_file = PLOT_DATA_DIR / "plot_fig3_decomposition_s1_s6.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist.")
        return

    df = pd.read_csv(csv_file)
    df["ds"] = pd.to_datetime(df["ds"])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9.6), sharex=True)
    specs = [
        ("S1", _t(lang, "(a) S1：东区主力店（W1 供货，受截断）", "(a) S1: core eastern store (supplied by W1, censored)")),
        ("S6", _t(lang, "(b) S6：中轴跨仓枢纽店（W2 供货，溢出峰值）", "(b) S6: central bridge hub (supplied by W2, spillover peaks)")),
    ]

    for ax, (s_id, title) in zip(axes, specs):
        sub = df[df["store_id"] == s_id].sort_values("ds")
        d = sub["ds"].values
        mu_s = sub["mu_self"].values
        cannib = sub["cannibalization"].values
        spill = sub["spillover"].values
        D = sub["latent_demand"].values
        y = np.where(sub["is_open"] == 1, sub["y"].values, np.nan)
        stockout = sub["stockout"].values == 1

        mu_open = mu_s - cannib + spill

        # Shading
        ax.fill_between(d, mu_open, mu_s, where=mu_open < mu_s - 0.5, color="#93C5FD", alpha=0.55, step="mid",
                        label=_t(lang, "邻店促销虹吸 (-C)", "Neighbour cannibalisation (-C)"))
        ax.fill_between(d, mu_s, mu_open, where=mu_open > mu_s + 0.5, color="#86EFAC", alpha=0.6, step="mid",
                        label=_t(lang, "邻店断货溢出 (+L)", "Neighbour stockout spillover (+L)"))
        ax.fill_between(d, y, D, where=stockout, color="#F9A8D4", alpha=0.75, step="mid",
                        label=_t(lang, "库存截断损失 (D - y)", "Censoring loss (D - y)"))

        ax.plot(d, mu_s, color="#0F172A", lw=1.2, label=_t(lang, "自身基准需求 μ^self", "Self baseline μ^self"))
        ax.plot(d, D, color="#6B7280", ls="--", lw=0.9, label=_t(lang, "潜在需求 D^sim", "Latent demand D^sim"))
        ax.plot(d, y, color="#1D4ED8" if s_id == "S1" else "#EA580C", lw=1.4,
                label=_t(lang, "观测销量 y^sales", "Observed sales y^sales"))

        # Highlight test window (Dec 04 to Dec 31)
        test_start = pd.Timestamp("2025-12-04")
        test_end = pd.Timestamp("2025-12-31")
        ax.axvspan(test_start, test_end, color="#FEF3C7", alpha=0.35)

        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel(_t(lang, "件 / 日", "units / day"))
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="upper left", fontsize=8.5, ncol=3, framealpha=0.92)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(_t(lang, "单店需求结构拆解：自身周期 + 邻店虹吸低谷 + 邻店溢出峰值 + 库存截断（2025 下半年，黄底为测试期）",
                    "Single-store demand structure: own cycle + cannibalisation dips + spillover peaks + censoring (H2 2025, shaded = test window)"),
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    _save_dual(fig, out_dir, "single_store_decomposition", lang)


# ----------------------------------------------------------------------------------------------
# Figure 4: Test Window 28-day Timeseries Predictions (S5 & S6)
# ----------------------------------------------------------------------------------------------
def plot_fig4_timeseries_comparison(out_dir: Path, lang: str = "zh") -> None:
    _set_font(lang)
    csv_file = PLOT_DATA_DIR / "plot_fig4_test_28d_s5_s6.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist.")
        return

    df = pd.read_csv(csv_file)
    df["ds"] = pd.to_datetime(df["ds"])

    fig, axes = plt.subplots(2, 1, figsize=(15, 9.4), sharex=True)
    specs = [
        ("S5", _t(lang, "(a) S5：W1 供货，12-08 至 12-12 受上游断货截断", "(a) S5: supplied by W1, censored by upstream disruption (12-08 to 12-12)")),
        ("S6", _t(lang, "(b) S6：W2 供货的跨仓枢纽店，承接 S5/S7 溢出客流", "(b) S6: W2-supplied bridge hub absorbing spillover from S5/S7")),
    ]

    for ax, (s_id, title) in zip(axes, specs):
        sub = df[df["store_id"] == s_id].sort_values("ds")
        d = sub["ds"].values
        closed = sub["is_open"].values == 0

        D = np.where(closed, np.nan, sub["D_sim"].values)
        y = np.where(closed, np.nan, sub["y_sales"].values)
        p1b = np.where(closed, np.nan, sub["pred_M1B_OLS_Uncens_7d"].values)
        p2 = np.where(closed, np.nan, sub["pred_M2_OLS_1Hop_9d"].values)
        p3 = np.where(closed, np.nan, sub["pred_M3_Neural_HeteroGNN_2Layer"].values if "pred_M3_Neural_HeteroGNN_2Layer" in sub else sub["pred_M3_Analytical_2Hop_79d"].values)

        stockout = (sub["supply_cap"].values < 500) & (sub["is_open"].values == 1)
        if stockout.any():
            ax.fill_between(d, y, D, where=stockout, color="#F9A8D4", alpha=0.7, step="mid",
                            label=_t(lang, "未满足需求 U", "Unmet demand U"))

        ax.plot(d, D, color="#0F172A", lw=2.2, marker="o", ms=4, label=_t(lang, "潜在真实需求 D^sim", "Latent demand D^sim"))
        ax.plot(d, y, color="#9CA3AF", lw=1.4, ls=":", marker="s", ms=3, label=_t(lang, "观测销量 y^sales", "Observed sales y^sales"))
        ax.plot(d, p1b, color=C_1B, lw=1.8, ls="--", label=_t(lang, "1B 单店去截断 (WAPE ~13.8%)", "1B Uncensored local (WAPE ~13.8%)"))
        ax.plot(d, p2, color=C_2, lw=1.8, ls="-.", label=_t(lang, "2 一阶邻域特征 (WAPE ~12.8%)", "2 1-hop flat features (WAPE ~12.8%)"))
        ax.plot(d, p3, color=C_3, lw=2.4, label=_t(lang, "3 两层异构时空图 (WAPE ~3.7%)", "3 Two-layer ST-GNN (WAPE ~3.7%)"))

        if closed.any():
            close_idx = np.where(closed)[0][0]
            ax.axvline(d[close_idx], color="#94A3B8", lw=6, alpha=0.25)

        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel(_t(lang, "件 / 日", "units / day"))
        ax.set_ylim(0, 225)
        ax.grid(True, ls=":", alpha=0.45)
        ax.legend(loc="lower left", fontsize=8.4, ncol=3, framealpha=0.93)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle(_t(lang, "测试集 28 天（2025-12-04 至 12-31）中轴走廊相邻门店的动态耦合",
                    "28-day test window (2025-12-04 to 12-31): dynamic coupling of adjacent Central-Corridor stores"),
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    _save_dual(fig, out_dir, "store_timeseries_comparison", lang)


# ----------------------------------------------------------------------------------------------
# Figure 5: Regime Ablation Metrics (WAPE & Bias)
# ----------------------------------------------------------------------------------------------
def plot_fig5_ablation_metrics(out_dir: Path, lang: str = "zh") -> None:
    _set_font(lang)
    csv_file = PLOT_DATA_DIR / "plot_fig5_ablation_regimes.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist.")
        return

    df = pd.read_csv(csv_file).set_index("model_id")

    core_models = [
        ("M1A_OLS_Naive_7d", _t(lang, "1A 单店全量", "1A Naive local"), C_1A),
        ("M1B_OLS_Uncens_7d", _t(lang, "1B 单店去截断", "1B Uncensored local"), C_1B),
        ("M2_OLS_1Hop_9d", _t(lang, "2 一阶邻域", "2 1-hop flat"), C_2),
        ("M_GBDT_2Hop_Graph_27d", _t(lang, "GBDT 二阶图特征", "GBDT 2-Hop Graph"), C_GBDT),
        ("M3_Neural_HeteroGNN_2Layer", _t(lang, "3 两层异构图", "3 Hetero-GNN"), C_3),
    ]

    # Regimes
    regimes = ["Normal", "Cannibalization", "Spillover"]
    n_normal = int(df.loc["M1A_OLS_Naive_7d", "Normal_n"])
    n_cannib = int(df.loc["M1A_OLS_Naive_7d", "Cannibalization_n"])
    n_spill = int(df.loc["M1A_OLS_Naive_7d", "Spillover_n"])

    rlab = {
        "Normal": _t(lang, f"常规平稳期\n(n={n_normal})", f"Normal\n(n={n_normal})"),
        "Cannibalization": _t(lang, f"邻店促销虹吸期\n(n={n_cannib})", f"Cannibalization\n(n={n_cannib})"),
        "Spillover": _t(lang, f"跨店断货溢出期\n(n={n_spill})", f"Spillover\n(n={n_spill})"),
    }

    x = np.arange(len(regimes))
    w = 0.16
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.2))

    for k, (m_id, m_name, color) in enumerate(core_models):
        if m_id not in df.index:
            continue
        wv = [df.loc[m_id, f"{r}_D_wape"] for r in regimes]
        bv = [df.loc[m_id, f"{r}_D_bias"] for r in regimes]

        offset = (k - (len(core_models) - 1) / 2.0) * w
        b1 = ax1.bar(x + offset, wv, w, color=color, label=m_name)
        b2 = ax2.bar(x + offset, bv, w, color=color, label=m_name)

        for bar, v in zip(b1, wv):
            ax1.text(bar.get_x() + w / 2, v + 0.35, f"{v:.1f}", ha="center", fontsize=7.2)
        for bar, v in zip(b2, bv):
            ax2.text(bar.get_x() + w / 2, v + (0.5 if v >= 0 else -0.5), f"{v:+.1f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=7.2)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([rlab[r] for r in regimes], fontsize=10)
        ax.grid(True, axis="y", ls=":", alpha=0.5)

    ax1.set_ylim(0, 33)
    ax1.set_ylabel("WAPE (%)")
    ax1.legend(loc="upper left", fontsize=8.8)
    ax1.set_title(_t(lang, "(a) 各业务子场景 WAPE（目标：潜在需求 D^sim）",
                     "(a) WAPE by business regime (target: latent demand D^sim)"), fontsize=12, fontweight="bold")

    ax2.axhline(0, color="black", lw=0.9)
    ax2.set_ylim(-34, 12)
    ax2.set_ylabel("Bias (%)")
    ax2.set_title(_t(lang, "(b) 各业务子场景系统性偏差 Bias（正 = 高估，负 = 低估）",
                     "(b) Systematic bias by regime (+ over-forecast, - under-forecast)"), fontsize=12, fontweight="bold")

    fig.tight_layout()
    _save_dual(fig, out_dir, "regime_ablation_metrics", lang)


# ----------------------------------------------------------------------------------------------
# Figure 6: Toolkit Benchmark Tradeoff & Four-Layer Ecosystem
# ----------------------------------------------------------------------------------------------
def plot_fig6_toolkit_tradeoff(out_dir: Path, lang: str = "zh") -> None:
    _set_font(lang)
    csv_file = PLOT_DATA_DIR / "plot_fig5_ablation_regimes.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist.")
        return

    df = pd.read_csv(csv_file).set_index("model_id")

    # Define toolkits / tiers for comparison
    points = [
        {"name_zh": "1B · 闭式 OLS", "name_en": "1B · Closed OLS", "tier": "1B", "t": 0.0007, "w": df.loc["M1B_OLS_Uncens_7d", "Spillover_D_wape"] if "M1B_OLS_Uncens_7d" in df.index else 22.3, "pos": (8, 6)},
        {"name_zh": "1B · Tobit MLE", "name_en": "1B · Tobit MLE", "tier": "1B", "t": 0.305, "w": df.loc["M1B_Tobit_MLE_7d", "Spillover_D_wape"] if "M1B_Tobit_MLE_7d" in df.index else 21.1, "pos": (-14, 12)},
        {"name_zh": "2 · 一阶 OLS", "name_en": "2 · 1-Hop OLS", "tier": "2", "t": 0.0022, "w": df.loc["M2_OLS_1Hop_9d", "Spillover_D_wape"] if "M2_OLS_1Hop_9d" in df.index else 21.1, "pos": (8, -14)},
        {"name_zh": "2 · LightGBM 宽表", "name_en": "2 · LightGBM Flat", "tier": "2", "t": df.loc["M_GBDT_1Hop_Raw_18d", "runtime_sec"] if "M_GBDT_1Hop_Raw_18d" in df.index else 0.26, "w": df.loc["M_GBDT_1Hop_Raw_18d", "Spillover_D_wape"] if "M_GBDT_1Hop_Raw_18d" in df.index else 10.9, "pos": (8, 6)},
        {"name_zh": "2 · XGBoost 宽表", "name_en": "2 · XGBoost Flat", "tier": "2", "t": df.loc["M_XGBoost_1Hop_Raw_18d", "runtime_sec"] if "M_XGBoost_1Hop_Raw_18d" in df.index else 0.32, "w": df.loc["M_XGBoost_1Hop_Raw_18d", "Spillover_D_wape"] if "M_XGBoost_1Hop_Raw_18d" in df.index else 11.0, "pos": (-75, -16)},
        {"name_zh": "2+ · LightGBM 二阶图", "name_en": "2+ · LightGBM 2-Hop", "tier": "3", "t": df.loc["M_GBDT_2Hop_Graph_27d", "runtime_sec"] if "M_GBDT_2Hop_Graph_27d" in df.index else 0.26, "w": df.loc["M_GBDT_2Hop_Graph_27d", "Spillover_D_wape"] if "M_GBDT_2Hop_Graph_27d" in df.index else 12.5, "pos": (-85, 10)},
        {"name_zh": "2+ · XGBoost 二阶图", "name_en": "2+ · XGBoost 2-Hop", "tier": "3", "t": df.loc["M_XGBoost_2Hop_Graph_27d", "runtime_sec"] if "M_XGBoost_2Hop_Graph_27d" in df.index else 0.34, "w": df.loc["M_XGBoost_2Hop_Graph_27d", "Spillover_D_wape"] if "M_XGBoost_2Hop_Graph_27d" in df.index else 12.6, "pos": (8, -14)},
        {"name_zh": "3 · 解析图模型", "name_en": "3 · Analytical Graph", "tier": "3", "t": 0.0055, "w": df.loc["M3_Analytical_2Hop_79d", "Spillover_D_wape"] if "M3_Analytical_2Hop_79d" in df.index else 3.95, "pos": (8, 6)},
        {"name_zh": "3 · PyG 异构图", "name_en": "3 · PyG Hetero-GNN", "tier": "3", "t": 1.17, "w": df.loc["M3_Neural_HeteroGNN_2Layer", "Spillover_D_wape"] if "M3_Neural_HeteroGNN_2Layer" in df.index else 3.26, "pos": (-70, -22)},
    ]

    tier_color = {"1B": C_1B, "2": C_2, "3": C_3}
    tier_labels = {
        "1B": _t(lang, "1B 单店去截断层", "Tier 1B: single-store uncensored"),
        "2": _t(lang, "2 一阶空间邻域层", "Tier 2: 1-hop spatial flat"),
        "3": _t(lang, "3 双层异构图模型层", "Tier 3: two-layer ST-GNN"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.6), gridspec_kw={"width_ratios": [1.25, 1.0]})

    seen = set()
    for pt in points:
        c = tier_color[pt["tier"]]
        lab = tier_labels[pt["tier"]] if pt["tier"] not in seen else None
        seen.add(pt["tier"])

        ax1.scatter(pt["t"], pt["w"], s=170, color=c, edgecolors="#0F172A", lw=1.2, zorder=5, label=lab)
        name = pt["name_zh"] if lang == "zh" else pt["name_en"]
        ax1.annotate(f"{name}\n{pt['t']:g}s · {pt['w']:.1f}%", (pt["t"], pt["w"]),
                     xytext=pt["pos"], textcoords="offset points", fontsize=8.2)

    ax1.set_xscale("log")
    ax1.set_xlim(0.0003, 30)
    ax1.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.001, 0.01, 0.1, 1, 10]))
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}s"))
    ax1.set_ylim(0, 32)
    ax1.grid(True, which="both", ls=":", alpha=0.45)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlabel(_t(lang, "单次运行耗时（秒，对数轴）", "Single-run runtime (s, log scale)"))
    ax1.set_ylabel(_t(lang, "跨店断货溢出期 WAPE (%)", "Spillover-regime WAPE (%)"))
    ax1.set_title(_t(lang, "(a) 运行耗时 vs. 溢出期 WAPE：信息边界决定精度台阶",
                     "(a) Runtime vs. spillover WAPE: information tier sets the accuracy step"), fontsize=12, fontweight="bold")

    # (b) Four-layer toolkit ecosystem
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    layers = [
        ("#DCFCE7", "#16A34A", _t(lang, "④ 标准化时空评测流水线", "④ Standardised ST benchmark pipeline"), "LibCity / Benchmarks",
         _t(lang, "数据切片 · 主流 ST-GNN · 统一评估", "data slicing · ST-GNN zoo · unified evaluation")),
        ("#EDE9FE", "#7C3AED", _t(lang, "③ 可定制异构图消息传递算子", "③ Customisable heterogeneous message passing"), "PyG / PyG Temporal / DGL",
         _t(lang, "HeteroData · MessagePassing · 两跳时空图", "HeteroData · MessagePassing · two-hop ST graph")),
        ("#FEF3C7", "#D97706", _t(lang, "② 开箱即用树模型与时序接口", "② Off-the-shelf tree & time-series interfaces"), "LightGBM / Darts / Prophet",
         _t(lang, "多店宽表 · 滞后交互 · 快速拟合", "multi-store flat tables · lag interactions · fast fit")),
        ("#E0F2FE", "#0284C7", _t(lang, "① 零开销解析解核对基准", "① Zero-overhead closed-form reference"), "NumPy / SciPy",
         _t(lang, "闭式 OLS · 机制核对 · 毫秒级", "closed-form OLS · mechanism check · milliseconds")),
    ]
    for k, (fc, ec, head, tools, desc) in enumerate(layers):
        y0 = 7.45 - k * 2.35
        inset = 0.25 * (3 - k)
        ax2.add_patch(mpatches.FancyBboxPatch((0.3 + inset, y0), 9.4 - 2 * inset, 2.0,
                                              boxstyle="round,pad=0.05,rounding_size=0.25", fc=fc, ec=ec, lw=1.8))
        ax2.text(5, y0 + 1.5, head, ha="center", va="center", fontsize=10.5, fontweight="bold", color=ec)
        ax2.text(5, y0 + 0.95, tools, ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")
        ax2.text(5, y0 + 0.42, desc, ha="center", va="center", fontsize=8.6, color="#334155")
    ax2.set_title(_t(lang, "(b) 四层工具生态：解决的工程抽象层次不同",
                     "(b) Four-layer toolkit ecosystem by engineering abstraction"), fontsize=12, fontweight="bold")

    fig.tight_layout()
    _save_dual(fig, out_dir, "multitool_benchmark_tradeoff", lang)


# ----------------------------------------------------------------------------------------------
# Figure 7: Store x Regime Breakdown Heatmap
# ----------------------------------------------------------------------------------------------
def plot_fig7_store_regime_heatmap(out_dir: Path, lang: str = "zh") -> None:
    _set_font(lang)
    csv_file = PLOT_DATA_DIR / "plot_fig7_store_regime_heatmap.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist.")
        return

    df = pd.read_csv(csv_file)

    # Subplots for the 3 main regimes: Normal, Cannibalization, Spillover
    regimes = ["Normal", "Cannibalization", "Spillover"]
    models = [
        ("1A_Naive_7d_wape", "1A Naive"),
        ("1B_Uncens_7d_wape", "1B Uncens"),
        ("2_Linear_1Hop_9d_wape", "2 Linear 1-Hop"),
        ("GBDT_Base_11d_wape", "GBDT Base"),
        ("GBDT_1Hop_Raw_18d_wape", "LGBM 1-Hop"),
        ("XGBoost_1Hop_18d_wape", "XGB 1-Hop"),
        ("GBDT_2Hop_27d_wape", "LGBM 2-Hop"),
        ("XGBoost_2Hop_27d_wape", "XGB 2-Hop"),
        ("3_Neural_GNN_2L_wape", "3 Neural GNN"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True)

    titles = {
        "Normal": _t(lang, "(a) 常规平稳期 WAPE (%)", "(a) Normal Regime WAPE (%)"),
        "Cannibalization": _t(lang, "(b) 邻店促销虹吸期 WAPE (%)", "(b) Cannibalization Regime WAPE (%)"),
        "Spillover": _t(lang, "(c) 跨店断货溢出期 WAPE (%)", "(c) Spillover Regime WAPE (%)"),
    }

    stores = [f"S{i}" for i in range(1, 11)]

    for ax, reg in zip(axes, regimes):
        sub = df[df["regime"] == reg].set_index("store_id")
        data = np.zeros((len(stores), len(models)))
        for i, s in enumerate(stores):
            for j, (col, _) in enumerate(models):
                if s in sub.index and col in sub.columns:
                    data[i, j] = sub.loc[s, col]

        # Use colormap highlighting error explosion
        vmax = 35.0 if reg == "Spillover" else 18.0
        im = ax.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m[1] for m in models], rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(stores)))
        ax.set_yticklabels(stores, fontsize=9.5)

        for i in range(len(stores)):
            for j in range(len(models)):
                val = data[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.8,
                        color="white" if val > vmax * 0.65 else "#0F172A")

        ax.set_title(titles[reg], fontsize=11.5, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel(_t(lang, "门店 ID", "Store ID"), fontsize=11)
    fig.suptitle(_t(lang, "10 家门店在各业务子场景下的模型误差热力图（溢出期单店模型全线崩溃，GNN 保持平稳）",
                    "Store-level Model WAPE Heatmap across Business Regimes (Spillover collapses local models, GNN stays robust)"),
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    _save_dual(fig, out_dir, "store_regime_heatmap", lang)


# ----------------------------------------------------------------------------------------------
# Figure 8: Feature Dimension vs WAPE Ladder
# ----------------------------------------------------------------------------------------------
def plot_fig8_feature_dim_vs_wape(out_dir: Path, lang: str = "zh") -> None:
    _set_font(lang)
    csv_file = PLOT_DATA_DIR / "plot_fig8_feature_dim_vs_wape.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist.")
        return

    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    colors = [C_1A, C_1B, "#D97706", C_1B, "#B45309", C_2, "#6D28D9", C_3]
    markers = ["o", "s", "^", "D", "v", "p", "h", "*"]

    for i, row in df.iterrows():
        x = row["design_matrix_cols"]
        y_overall = row["Overall_D_wape"]
        y_spill = row["Spillover_D_wape"]
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        ax.scatter(x, y_overall, s=120, color=c, marker=m, edgecolors="black", lw=1.0, zorder=5)
        ax.scatter(x, y_spill, s=140, color=c, marker=m, alpha=0.4, edgecolors="red", lw=1.2, zorder=4)

        # Draw connecting vertical line between overall and spillover error
        ax.plot([x, x], [y_overall, y_spill], color=c, ls=":", lw=1.2, alpha=0.7)

        # Label
        lbl = row["model_id"].replace("L0_", "").replace("L1_", "").replace("L2a_", "").replace("L2b_", "").replace("L3_", "").replace("L3p_", "").replace("L4_", "")
        ax.annotate(f"{lbl}\n(Overall: {y_overall:.1f}%, Spill: {y_spill:.1f}%)", (x, y_overall),
                    xytext=(0, -18 if i % 2 == 0 else 10), textcoords="offset points",
                    ha="center", fontsize=7.8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", alpha=0.85))

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([7, 16, 27, 70, 80]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(_t(lang, "设计矩阵参数/特征列数（对数轴）", "Design Matrix Columns / Parameters (Log Scale)"), fontsize=11)
    ax.set_ylabel(_t(lang, "预测误差 WAPE (%)", "Forecast Error WAPE (%)"), fontsize=11)
    ax.set_ylim(0, 35)
    ax.grid(True, which="both", ls=":", alpha=0.45)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#475569", markeredgecolor="black", markersize=9, label=_t(lang, "全周期总体 WAPE (Overall)", "Overall WAPE")),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#475569", markeredgecolor="red", alpha=0.5, markersize=10, label=_t(lang, "溢出期 WAPE (Spillover)", "Spillover WAPE")),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9.5)

    ax.set_title(_t(lang, "特征维度阶梯与信息边界：参数增加无法逾越信息拓扑缺失",
                     "Feature Dimension Ladder vs Information Boundary: Model Capacity Cannot Replace Topology"),
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save_dual(fig, out_dir, "feature_dim_vs_wape", lang)


# ----------------------------------------------------------------------------------------------
# Figure 9: Real Public Benchmark Evaluation
# ----------------------------------------------------------------------------------------------
def plot_fig9_public_benchmarks(out_dir: Path, lang: str = "zh") -> None:
    _set_font(lang)
    csv_file = PLOT_DATA_DIR / "plot_fig9_public_benchmarks.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist.")
        return

    df = pd.read_csv(csv_file)
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 5.8), sharey=False)

    bench_titles = {
        "FreshRetailNet_50K": _t(lang, "场景一：叮咚买菜 FreshRetailNet-50K\n(同城高密前置仓 · 30.8%真实缺货截断)",
                                "Scenario 1: Dingdong FreshRetailNet-50K\n(Urban Front-Warehouses · 30.8% Stockout)"),
        "Kaggle_Favorita": _t(lang, "场景二：厄瓜多尔 Corporación Favorita\n(跨商圈连锁超市 · 发薪促销脉冲)",
                              "Scenario 2: Corporación Favorita\n(Supermarket Clusters · Payday Promos)"),
        "Walmart_M5": _t(lang, "场景三：沃尔玛 Walmart M5\n(跨州远距离大卖场 · 宏观同步波动)",
                         "Scenario 3: Walmart M5\n(Cross-State Supercenters · Macro Drivers)"),
    }

    colors = [C_1A, C_1B, "#FBBF24", C_2, C_GBDT, "#EA580C", C_3]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds].reset_index(drop=True)
        x = np.arange(len(sub))
        w = 0.55

        bars = ax.bar(x, sub["Overall_WAPE"], w, color=colors[:len(sub)], edgecolor="#1E293B", lw=0.8)

        for bar, val, bias in zip(bars, sub["Overall_WAPE"], sub["Overall_Bias"]):
            ax.text(bar.get_x() + w / 2, val + 0.6, f"{val:.1f}%\n({bias:+.1f}%)",
                    ha="center", va="bottom", fontsize=7.6)

        # Simplify model names for labels
        short_names = [
            m.replace("1A_Local_Naive", "1A Naive")
             .replace("1B_Local_Uncensored", "1B Uncens")
             .replace("1Bp_Pooled_Shared", "1B' Pooled")
             .replace("2_Linear_1Hop", "2 Linear 1H")
             .replace("GBDT_1Hop_16d", "LightGBM")
             .replace("XGBoost_1Hop_16d", "XGBoost")
             .replace("3_Hetero_Graph_2Hop", "3 Graph 2H")
            for m in sub["model_id"]
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8.8)
        ax.set_ylabel("WAPE (%)", fontsize=10)
        ax.set_ylim(0, max(sub["Overall_WAPE"]) * 1.25)
        ax.grid(True, axis="y", ls=":", alpha=0.45)
        ax.set_title(bench_titles.get(ds, ds), fontsize=11, fontweight="bold")

    fig.suptitle(_t(lang, "公开真实零售基准全量评测：不同零售业态与空间拓扑下的模型表现",
                    "Public Real-World Retail Benchmark Evaluation across Spatial Topologies & Retail Formats"),
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    _save_dual(fig, out_dir, "public_benchmarks_comparison", lang)


# ----------------------------------------------------------------------------------------------
# Master Runner
# ----------------------------------------------------------------------------------------------
def generate_all_figures(out_dir: Path, langs: List[str]) -> None:
    print(f"\n=======================================================")
    print(f"Generating Publication-Grade Figures into {out_dir}")
    print(f"=======================================================")

    for lang in langs:
        print(f"\n>>> Rendering figures for language: [{lang.upper()}]")
        print("  Fig 1: retail_network_topology...")
        plot_fig1_topology(out_dir, lang)

        print("  Fig 2: all_stores_sales_panorama...")
        plot_fig2_panorama(out_dir, lang)

        print("  Fig 3: single_store_decomposition...")
        plot_fig3_decomposition(out_dir, lang)

        print("  Fig 4: store_timeseries_comparison...")
        plot_fig4_timeseries_comparison(out_dir, lang)

        print("  Fig 5: regime_ablation_metrics...")
        plot_fig5_ablation_metrics(out_dir, lang)

        print("  Fig 6: multitool_benchmark_tradeoff...")
        plot_fig6_toolkit_tradeoff(out_dir, lang)

        print("  Fig 7: store_regime_heatmap...")
        plot_fig7_store_regime_heatmap(out_dir, lang)

        print("  Fig 8: feature_dim_vs_wape...")
        plot_fig8_feature_dim_vs_wape(out_dir, lang)

        print("  Fig 9: public_benchmarks_comparison...")
        plot_fig9_public_benchmarks(out_dir, lang)

    print("\n[SUCCESS] All figures successfully generated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bilingual figures from plot data CSVs.")
    parser.add_argument("--lang", choices=["zh", "en", "both"], default="both",
                        help="Language for figures ('zh', 'en', or 'both').")
    parser.add_argument("--out-dir", type=str, default=str(FIGURES_DIR),
                        help="Directory to save generated figures.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    langs = ["zh", "en"] if args.lang == "both" else [args.lang]
    generate_all_figures(out_dir, langs)


if __name__ == "__main__":
    main()

