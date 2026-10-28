#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilingual (Chinese & English) figure generator for Chapter 4 of the blog post
=============================================================================

https://cuicaihao.github.io/zh/posts/2026-10-24-fsm-stochastic-gnn-retail-forecast/

Figure 1 (Sec 4.1) retail_network_topology        (a) heterogeneous topology  (b) spatial random-walk matrix P^spa
Figure 2 (Sec 4.3) all_stores_sales_panorama      10-store daily sales vs. latent demand, H2 2025 (184 days)
Figure 3 (Sec 4.3) single_store_decomposition     S1 (censored core store) vs. S6 (spillover bridge store)
Figure 4 (Sec 4.4) store_timeseries_comparison    28-day test window: (a) S5 censored  (b) S6 spillover hub
Figure 5 (Sec 4.5) regime_ablation_metrics        (a) regime WAPE  (b) regime Bias
Figure 6 (Sec 4.6) multitool_benchmark_tradeoff   (a) runtime vs. spillover WAPE  (b) four-layer toolkit ecosystem

Each figure is written as:
  figures/<stem>.png, figures/<stem>.zh.webp      (Chinese)
  figures/<stem>_en.png, figures/<stem>.en.webp   (English)
The *.zh.webp / *.en.webp files use the same names as the Hugo page bundle of the blog post.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import logging
from typing import Any, Dict

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

ZH_FONTS = ["Hiragino Sans GB", "PingFang SC", "Heiti SC", "Heiti TC", "Arial Unicode MS",
            "Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
EN_FONTS = ["DejaVu Sans", "Arial", "Helvetica"]

C_W1, C_W2 = "#1D4ED8", "#C2410C"          # warehouses
C_S_W1, C_S_W2 = "#60A5FA", "#FB923C"      # stores by supplier
C_1A, C_1B, C_2, C_3 = "#EF4444", "#F59E0B", "#8B5CF6", "#10B981"


def _set_font(lang: str) -> None:
    plt.rcParams["font.sans-serif"] = ZH_FONTS if lang == "zh" else EN_FONTS
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def _t(lang: str, zh: str, en: str) -> str:
    return zh if lang == "zh" else en


def _save_dual(fig: plt.Figure, out_dir: str, stem: str, lang: str, dpi: int = 160) -> None:
    """Save as PNG (repo preview) and WebP (blog page-bundle name)."""
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{stem}.png" if lang == "zh" else f"{stem}_en.png")
    webp_path = os.path.join(out_dir, f"{stem}.{lang}.webp")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(png_path) as im:
        im.convert("RGB").save(webp_path, "WEBP", quality=88, method=6)


# ----------------------------------------------------------------------------------------------
# Figure 1
# ----------------------------------------------------------------------------------------------
def plot_fig1_topology(sim: Dict[str, Any], out_dir: str, lang: str = "zh") -> None:
    _set_font(lang)
    sc, wc, wa = sim["store_coords"], sim["warehouse_coords"], sim["w_assign"]
    A, P, ids, N = sim["A_spatial"], sim["P_spatial"], sim["store_ids"], sim["N_stores"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

    for i in range(N):  # supply edges E^sup (dashed)
        w = wa[i]
        ax1.plot([wc[w, 0], sc[i, 0]], [wc[w, 1], sc[i, 1]], ls="--", lw=1.2,
                 color=C_W1 if w == 0 else C_W2, alpha=0.45, zorder=1)
    for i in range(N):  # spatial edges E^spa (solid, width ~ A_ij)
        for j in range(i + 1, N):
            if A[i, j] > 0:
                ax1.plot([sc[i, 0], sc[j, 0]], [sc[i, 1], sc[j, 1]], color="#475569",
                         lw=1.0 + 3.5 * A[i, j], alpha=0.6, zorder=2)
    for w, (x, y) in enumerate(wc):
        ax1.scatter(x, y, s=520, marker="s", color=C_W1 if w == 0 else C_W2, edgecolors="black", lw=1.5, zorder=5)
        ax1.text(x, y, f"W{w + 1}", color="white", fontsize=10.5, fontweight="bold", ha="center", va="center", zorder=6)
    for i, (x, y) in enumerate(sc):
        bridge = i in (3, 5)
        ax1.scatter(x, y, s=360 if bridge else 290, color=C_S_W1 if wa[i] == 0 else C_S_W2,
                    edgecolors="#DC2626" if bridge else "#1E293B", lw=2.6 if bridge else 1.2, zorder=5)
        ax1.text(x, y, ids[i], fontsize=9, fontweight="bold", ha="center", va="center", zorder=6)

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
    ax2.set_xticks(range(N), ids, fontsize=9)
    ax2.set_yticks(range(N), ids, fontsize=9)
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
# Figure 2
# ----------------------------------------------------------------------------------------------
def plot_fig2_panorama(sim: Dict[str, Any], out_dir: str, lang: str = "zh") -> None:
    _set_font(lang)
    dates = sim["dates"]
    start = int(np.where(dates == pd.Timestamp("2025-07-01"))[0][0])  # H2 2025: 184 days
    d = dates[start:]
    D, y, so, q, O = (sim["D_sim"][start:], sim["y_sales"][start:], sim["stockout_flag"][start:],
                      sim["q_it"][start:], sim["O_t"][start:])
    ids, wa = sim["store_ids"], sim["w_assign"]

    fig, axes = plt.subplots(5, 2, figsize=(17, 13.5), sharex=True, sharey=True)
    for i, ax in enumerate(axes.T.flatten()):
        bridge = i in (3, 5)
        promo = q[:, i] > 0
        ax.fill_between(d, 0, 1, where=promo, transform=ax.get_xaxis_transform(), color="#FEF08A", alpha=0.45, lw=0, step="mid")
        ax.plot(d, D[:, i], color="#6B7280", ls="--", lw=0.9, label=_t(lang, "潜在需求 D^sim", "Latent demand D^sim"))
        ax.plot(d, np.where(O == 1, y[:, i], np.nan), color="#EA580C" if bridge else "#1D4ED8", lw=1.2,
                label=_t(lang, "观测销量 y^sales", "Observed sales y^sales"))
        m = so[:, i] == 1
        ax.scatter(d[m], y[m, i], marker="v", color="#DC2626", s=22, zorder=5, label=_t(lang, "缺货截断", "Stockout censoring"))
        role = _t(lang, "跨仓枢纽", "bridge") if bridge else ""
        ax.set_title(f"{ids[i]} · W{wa[i] + 1}" + (f" · {role}" if role else "")
                     + f"  ({_t(lang, '日均销量', 'mean sales')} {y[O == 1, i].mean():.0f})",
                     fontsize=10, fontweight="bold", loc="left", color="#C2410C" if bridge else "#0F172A")
        ax.grid(True, ls=":", alpha=0.4)
    axes[0, 0].legend(handles=axes[0, 0].get_legend_handles_labels()[0] + [mpatches.Patch(color="#FEF08A", label=_t(lang, "促销窗口", "Promotion window"))],
                      loc="upper left", fontsize=8, ncol=4, framealpha=0.92)
    for ax in axes[-1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle(_t(lang, "10 家门店日度观测销量与潜在需求（2025-07-01 至 2025-12-31，共 184 天）",
                    "Daily observed sales vs. latent demand across 10 stores (2025-07-01 to 2025-12-31, 184 days)"),
                 fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout()
    _save_dual(fig, out_dir, "all_stores_sales_panorama", lang)


# ----------------------------------------------------------------------------------------------
# Figure 3
# ----------------------------------------------------------------------------------------------
def plot_fig3_decomposition(sim: Dict[str, Any], out_dir: str, lang: str = "zh") -> None:
    _set_font(lang)
    dates = sim["dates"]
    start = int(np.where(dates == pd.Timestamp("2025-07-01"))[0][0])
    d = dates[start:]
    O = sim["O_t"][start:]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9.6), sharex=True)
    specs = [
        (0, _t(lang, "(a) S1：东区主力店（W1 供货，受截断）", "(a) S1: core eastern store (supplied by W1, censored)")),
        (5, _t(lang, "(b) S6：中轴跨仓枢纽店（W2 供货，溢出峰值）", "(b) S6: central bridge hub (supplied by W2, spillover peaks)")),
    ]
    for ax, (s, title) in zip(axes, specs):
        mu_s = sim["mu_self"][start:, s]
        mu_o = sim["mu_open"][start:, s]
        D = sim["D_sim"][start:, s]
        y = np.where(O == 1, sim["y_sales"][start:, s], np.nan)
        ax.fill_between(d, mu_o, mu_s, where=mu_o < mu_s - 0.5, color="#93C5FD", alpha=0.55, step="mid",
                        label=_t(lang, "邻店促销虹吸 (-C)", "Neighbour cannibalisation (-C)"))
        ax.fill_between(d, mu_s, mu_o, where=mu_o > mu_s + 0.5, color="#86EFAC", alpha=0.6, step="mid",
                        label=_t(lang, "邻店断货溢出 (+L)", "Neighbour stockout spillover (+L)"))
        ax.fill_between(d, y, D, where=sim["stockout_flag"][start:, s] == 1, color="#F9A8D4", alpha=0.75, step="mid",
                        label=_t(lang, "库存截断损失 (D - y)", "Censoring loss (D - y)"))
        ax.plot(d, mu_s, color="#0F172A", lw=1.2, label=_t(lang, "自身基准需求 μ^self", "Self baseline μ^self"))
        ax.plot(d, D, color="#6B7280", ls="--", lw=0.9, label=_t(lang, "潜在需求 D^sim", "Latent demand D^sim"))
        ax.plot(d, y, color="#1D4ED8" if s == 0 else "#EA580C", lw=1.4, label=_t(lang, "观测销量 y^sales", "Observed sales y^sales"))
        ax.axvspan(dates[sim["train_end"]], dates[-1], color="#FEF3C7", alpha=0.35)
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
# Figure 4
# ----------------------------------------------------------------------------------------------
def plot_fig4_timeseries_comparison(sim: Dict[str, Any], ablation: Dict[str, Any], metrics: Dict[str, Any],
                                    out_dir: str, lang: str = "zh") -> None:
    _set_font(lang)
    te = sim["train_end"]
    d = sim["dates"][te:]
    O = sim["O_t"][te:]
    wm = {k: metrics["overall"][k]["D_sim"]["WMAPE"] for k in ("1B", "2", "3")}

    fig, axes = plt.subplots(2, 1, figsize=(15, 9.4), sharex=True)
    specs = [
        (4, _t(lang, "(a) S5：W1 供货，12-08 至 12-12 受上游断货截断", "(a) S5: supplied by W1, censored by upstream disruption (12-08 to 12-12)")),
        (5, _t(lang, "(b) S6：W2 供货的跨仓枢纽店，承接 S5/S7 溢出客流", "(b) S6: W2-supplied bridge hub absorbing spillover from S5/S7")),
    ]
    for ax, (s, title) in zip(axes, specs):
        closed = O == 0  # Christmas Day: store closed, no forecast is evaluated
        D = np.where(closed, np.nan, sim["D_sim"][te:, s])
        y = np.where(closed, np.nan, sim["y_sales"][te:, s])
        so = sim["stockout_flag"][te:, s] == 1
        if so.any():
            ax.fill_between(d, y, D, where=so, color="#F9A8D4", alpha=0.7, step="mid",
                            label=_t(lang, "未满足需求 U", "Unmet demand U"))
        ax.plot(d, D, color="#0F172A", lw=2.2, marker="o", ms=4, label=_t(lang, "潜在真实需求 D^sim", "Latent demand D^sim"))
        ax.plot(d, y, color="#9CA3AF", lw=1.4, ls=":", marker="s", ms=3, label=_t(lang, "观测销量 y^sales", "Observed sales y^sales"))
        ax.plot(d, np.where(closed, np.nan, ablation["pred_1b"][te:, s]), color=C_1B, lw=1.8, ls="--",
                label=_t(lang, f"1B 单店去截断 (全网 WAPE {wm['1B']:.2f}%)", f"1B Uncensored local (network WAPE {wm['1B']:.2f}%)"))
        ax.plot(d, np.where(closed, np.nan, ablation["pred_m2_demand"][te:, s]), color=C_2, lw=1.8, ls="-.",
                label=_t(lang, f"2 一阶邻域特征 (全网 WAPE {wm['2']:.2f}%)", f"2 1-hop flat features (network WAPE {wm['2']:.2f}%)"))
        ax.plot(d, np.where(closed, np.nan, ablation["pred_m3_demand"][te:, s]), color=C_3, lw=2.4,
                label=_t(lang, f"3 两层解析图 (全网 WAPE {wm['3']:.2f}%)", f"3 Two-layer analytical graph (network WAPE {wm['3']:.2f}%)"))
        ax.axvline(d[np.argmax(closed)], color="#94A3B8", lw=6, alpha=0.25)
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
# Figure 5
# ----------------------------------------------------------------------------------------------
def plot_fig5_ablation_metrics(metrics: Dict[str, Any], out_dir: str, lang: str = "zh") -> None:
    _set_font(lang)
    reg = metrics["regimes"]
    models = ["1A", "1B", "2", "3"]
    names = {
        "1A": _t(lang, "1A 单店全量", "1A Naive local"),
        "1B": _t(lang, "1B 单店去截断", "1B Uncensored local"),
        "2": _t(lang, "2 一阶邻域特征", "2 1-hop flat"),
        "3": _t(lang, "3 两层解析图", "3 Analytical graph"),
    }
    colors = {"1A": C_1A, "1B": C_1B, "2": C_2, "3": C_3}
    regimes = ["Normal", "Cannibalization", "Spillover"]
    rlab = {
        "Normal": _t(lang, f"常规平稳期\n(n={reg['1A']['Normal']['n']})", f"Normal\n(n={reg['1A']['Normal']['n']})"),
        "Cannibalization": _t(lang, f"邻店促销虹吸期\n(n={reg['1A']['Cannibalization']['n']})", f"Cannibalization\n(n={reg['1A']['Cannibalization']['n']})"),
        "Spillover": _t(lang, f"跨店断货溢出期\n(n={reg['1A']['Spillover']['n']})", f"Spillover\n(n={reg['1A']['Spillover']['n']})"),
    }
    x = np.arange(len(regimes))
    w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))
    for k, m in enumerate(models):
        wv = [reg[m][r]["WMAPE"] for r in regimes]
        bv = [reg[m][r]["Bias"] for r in regimes]
        b1 = ax1.bar(x + (k - 1.5) * w, wv, w, color=colors[m], label=names[m])
        b2 = ax2.bar(x + (k - 1.5) * w, bv, w, color=colors[m], label=names[m])
        for bar, v in zip(b1, wv):
            ax1.text(bar.get_x() + w / 2, v + 0.4, f"{v:.2f}", ha="center", fontsize=7.6)
        for bar, v in zip(b2, bv):
            ax2.text(bar.get_x() + w / 2, v + (0.6 if v >= 0 else -0.6), f"{v:+.2f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=7.6)
    for ax in (ax1, ax2):
        ax.set_xticks(x, [rlab[r] for r in regimes], fontsize=10)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax1.set_ylim(0, 33)
    ax1.set_ylabel("WAPE (%)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(_t(lang, "(a) 各业务子场景 WAPE（目标：潜在需求 D^sim）", "(a) WAPE by business regime (target: latent demand D^sim)"),
                  fontsize=12, fontweight="bold")
    ax2.axhline(0, color="black", lw=0.9)
    ax2.set_ylim(-34, 12)
    ax2.set_ylabel("Bias (%)")
    ax2.set_title(_t(lang, "(b) 各业务子场景系统性偏差 Bias（正 = 高估，负 = 低估）", "(b) Systematic bias by regime (+ over-forecast, - under-forecast)"),
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_dual(fig, out_dir, "regime_ablation_metrics", lang)


# ----------------------------------------------------------------------------------------------
# Figure 6
# ----------------------------------------------------------------------------------------------
def plot_fig6_toolkit_tradeoff(scorecard: pd.DataFrame, out_dir: str, lang: str = "zh") -> None:
    _set_font(lang)
    tier_color = {"1B · 单店时序": C_1B, "2 · 扁平空间特征": C_2, "3 · 双层时空图": C_3}
    tier_en = {"1B · 单店时序": "Tier 1B: single-store", "2 · 扁平空间特征": "Tier 2: 1-hop flat", "3 · 双层时空图": "Tier 3: two-hop graph"}
    tier_zh = {"1B · 单店时序": "1B 单店", "2 · 扁平空间特征": "2 一阶", "3 · 双层时空图": "3 两跳图"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    seen = set()
    offsets = {
        "1b_ols": (8, 6), "1b_prophet": (-14, 12), "1b_darts": (-78, -4), "2_ols": (8, -14), "2_prophet": (6, -30),
        "3_analytical": (8, 6), "3_pyg": (-70, 10), "3_libcity": (8, -28),
    }
    for _, r in scorecard.iterrows():
        c = tier_color[r["tier"]]
        lab = None
        if r["tier"] not in seen:
            lab = tier_zh[r["tier"]] if lang == "zh" else tier_en[r["tier"]]
            seen.add(r["tier"])
        ax1.scatter(r["time_sec"], r["spillover_wmape"], s=170, color=c, edgecolors="#0F172A", lw=1.2, zorder=5, label=lab)
        name = r["toolkit_zh"] if lang == "zh" else r["toolkit_en"]
        name = name.split(" · ", 1)[1]
        ax1.annotate(f"{name}\n{r['time_sec']:g}s · {r['spillover_wmape']:.2f}%", (r["time_sec"], r["spillover_wmape"]),
                     xytext=offsets.get(r["key"], (8, 6)), textcoords="offset points", fontsize=8.2)
    ax1.set_xscale("log")
    ax1.set_xlim(0.002, 200)
    ax1.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.001, 0.01, 0.1, 1, 10, 100]))
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}s"))
    ax1.set_ylim(0, 27)
    ax1.grid(True, which="both", ls=":", alpha=0.45)
    ax1.legend(loc="center right", fontsize=9)
    ax1.set_xlabel(_t(lang, "单次运行耗时（秒，对数轴）", "Single-run runtime (s, log scale)"))
    ax1.set_ylabel(_t(lang, "跨店断货溢出期 WAPE (%)", "Spillover-regime WAPE (%)"))
    ax1.set_title(_t(lang, "(a) 运行耗时 vs. 溢出期 WAPE：信息边界决定精度台阶",
                     "(a) Runtime vs. spillover WAPE: information tier sets the accuracy step"), fontsize=12, fontweight="bold")

    # (b) Four-layer toolkit ecosystem
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    layers = [
        ("#DCFCE7", "#16A34A", _t(lang, "④ 标准化时空评测流水线", "④ Standardised ST benchmark pipeline"), "LibCity",
         _t(lang, "数据切片 · 主流 ST-GNN · 统一评估", "data slicing · ST-GNN zoo · unified evaluation")),
        ("#EDE9FE", "#7C3AED", _t(lang, "③ 可定制异构图消息传递算子", "③ Customisable heterogeneous message passing"), "PyG / PyG Temporal / DGL",
         _t(lang, "HeteroData · MessagePassing · 两跳时空图", "HeteroData · MessagePassing · two-hop ST graph")),
        ("#FEF3C7", "#D97706", _t(lang, "② 开箱即用时序工程接口", "② Off-the-shelf time-series interfaces"), "Prophet / Darts",
         _t(lang, "趋势 + 傅里叶季节 + add_regressor · TimeSeries", "trend + Fourier seasonality + add_regressor · TimeSeries")),
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
    ax2.set_title(_t(lang, "(b) 四层工具生态：解决的工程抽象层次不同", "(b) Four-layer toolkit ecosystem by engineering abstraction"),
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_dual(fig, out_dir, "multitool_benchmark_tradeoff", lang)


def generate_all_figures(sim: Dict[str, Any], ablation: Dict[str, Any], metrics: Dict[str, Any], out_dir: str) -> None:
    """Generate all six Chapter 4 figures in Chinese and English (PNG + WebP, 24 files)."""
    for lang in ("zh", "en"):
        plot_fig1_topology(sim, out_dir, lang)
        plot_fig2_panorama(sim, out_dir, lang)
        plot_fig3_decomposition(sim, out_dir, lang)
        plot_fig4_timeseries_comparison(sim, ablation, metrics, out_dir, lang)
        plot_fig5_ablation_metrics(metrics, out_dir, lang)
        plot_fig6_toolkit_tradeoff(metrics["scorecard"], out_dir, lang)


if __name__ == "__main__":
    from retail_gnn_benchmark import (
        BASE_DIR,
        compute_chapter4_metrics,
        fit_ablation_models,
        fit_toolkit_benchmark,
        run_simulation,
    )

    sim_data = run_simulation(seed=42)
    ablation_data = fit_ablation_models(sim_data)
    toolkit_data = fit_toolkit_benchmark(sim_data, ablation_data)
    metrics_data = compute_chapter4_metrics(sim_data, ablation_data, toolkit_data)
    figures_dir = os.path.join(BASE_DIR, "figures")
    generate_all_figures(sim_data, ablation_data, metrics_data, figures_dir)
    print(f"Generated 6 bilingual figures (24 files) in {figures_dir}")
