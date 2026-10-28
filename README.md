# Retail Network Demand Forecasting: From Single-Store Baselines to Heterogeneous Spatio-Temporal GNNs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Blog Post](https://img.shields.io/badge/Blog-cuicaihao.github.io-brightgreen.svg)](https://cuicaihao.github.io/zh/posts/2026-10-24-fsm-stochastic-gnn-retail-forecast/)
[![Technical Report](https://img.shields.io/badge/Technical_Report-experimental--report.md-blue.svg)](experimental-report.md)

本项目是技术深度长文 **《用图神经网络做零售需求预测：数学原理、受控实验与生产落地边界》**（[中文博文](https://cuicaihao.github.io/zh/posts/2026-10-24-fsm-stochastic-gnn-retail-forecast/) / [English Post](https://cuicaihao.github.io/posts/2026-10-24-fsm-stochastic-gnn-retail-forecast/)）与姊妹篇《[用 Prophet 做零售需求预测](https://cuicaihao.github.io/zh/posts/2026-10-01-prophet-retail-demand-forecast/)》配套的开源科学实验工程套件。

项目聚焦多仓多店零售网络中的核心冲突——**近邻促销负向虹吸**与**跨仓断货正向溢出**，构建了严格无信息泄漏的受控仿真沙盒与四大递进实验，并在三大真实公开工业级基准（**叮咚买菜 FreshRetailNet-50K**、**Kaggle Corporación Favorita**、**Walmart M5**）上进行了端到端实测验证。

---

## 目录

- [Retail Network Demand Forecasting: From Single-Store Baselines to Heterogeneous Spatio-Temporal GNNs](#retail-network-demand-forecasting-from-single-store-baselines-to-heterogeneous-spatio-temporal-gnns)
  - [目录](#目录)
  - [一、核心结论速览 (Executive Summary)](#一核心结论速览-executive-summary)
  - [二、现代模块化仓库架构](#二现代模块化仓库架构)
  - [三、快速开始 (Quickstart)](#三快速开始-quickstart)
    - [1. 环境安装](#1-环境安装)
    - [2. 自动下载三大真实公开数据](#2-自动下载三大真实公开数据)
    - [3. 一键运行全量实验套件与图表生成](#3-一键运行全量实验套件与图表生成)
    - [4. 单独运行指定子实验](#4-单独运行指定子实验)
    - [5. 启动交互式 Notebook](#5-启动交互式-notebook)
  - [四、详细技术实验报告与业务故事](#四详细技术实验报告与业务故事)
  - [五、数据产物持久化与绘图解耦设计](#五数据产物持久化与绘图解耦设计)
  - [六、历史版本归档说明 (Archive)](#六历史版本归档说明-archive)
  - [七、引用与关联阅读](#七引用与关联阅读)

---

## 一、核心结论速览 (Executive Summary)

![零售网络异构图与空间转移矩阵](figures/fig1_retail_network_topology.png)

1. **信息边界决定精度上限，盲目调包无法突破物理盲区**：
   在单店信息孤岛下，不论使用毫秒级闭式 OLS、还是花费数十秒拟合的 Meta Prophet 或 Darts，常规平稳期虽可达 3.7% WAPE，但在跨店断货溢出期均发生 **-22% 的严重低估漏报**。只有打通“仓库 $\to$ 邻店 $\to$ 本店”的两跳异构拓扑，溢出期误差才能骤降至 **3.26%–3.95%**。

![各业务子场景 WAPE 与系统性偏差 Bias 对比](figures/fig5_regime_ablation_metrics.png)

1. **树模型双雄是工业级强基线，能自主吃透大部分非线性组合**：
   当把“本店信息 + 邻居大促 + 邻居上游仓断货比例”并排铺平为 18 维宽表交给 GBDT（LightGBM 与 XGBoost）时，决策树无需人工显式相乘，仅凭分支判断即可将溢出期误差从 21.11% 压缩到 **10.38%（XGBoost）与 10.91%（LightGBM）**，自主追回近六成差距。两款引擎学出的前列特征高度自洽。

![三大真实公开基准评测对比](figures/fig9_public_benchmarks_comparison.png)

1. **真实世界的“三阶梯落地定律”与工具包选型**：
   - **高密生鲜网格（FreshRetailNet-50K）**：真实营业期缺货达 30.8%，去截断模型与全网池化（27.02%）表现最稳，树模型中 XGBoost（32.17%）优于 LightGBM（36.20%），交界仓上图网络保持领先；
   - **连锁大型商超（Favorita）**：缺失显式缺货记录，核心驱动力转为发薪日大促脉冲，**XGBoost (12.11%) 斩获全场第一，与图模型 (12.99%) 和 LightGBM (13.19%) 并驾齐驱**；
   - **远距离跨州卖场（Walmart M5）**：跨州无跑单客流，本地日历占绝对主导，**极简的一阶线性模型仅用 0.0006 秒（13.95%）全面反超复杂的树模型与图模型**。

---

## 二、现代模块化仓库架构

```text
retail-gnn-demand-forecast/
├── README.md                                  # 项目概览与快速启动指南
├── experimental-report.md                     # 完整的技术报告、数学规范与全量数据表
├── pyproject.toml                             # uv / pip 现代化项目依赖配置
├── requirements.txt                           # 核心依赖清单
├── retail_gnn_demand_forecast.ipynb           # 轻量化（29KB）交互式复现 Notebook
│
├── src/                                       # 核心算法与数据处理引擎
│   ├── data_generator.py                      # 4 层耦合零售网络沙盒（FSM / 马尔可夫双仓 / 空间转移）
│   ├── feature_builders.py                    # 统一特征构造器（L0..L4、GBDT 11d/18d/27d、GNN 张量）
│   ├── models_linear.py                       # 线性正规方程求解器与 Right-Censored Tobit MLE
│   ├── models_prophet.py                      # Prophet 时间序列加性模型（0-Hop / 1-Hop / 2-Hop 空间阶梯）
│   ├── models_tree.py                         # 树模型引擎（支持原生 LightGBM、XGBoost 与纯 NumPy 降级）
│   ├── models_gnn.py                          # 端到端两层异构图神经网络（纯 NumPy Adam 优化，5 随机种子）
│   ├── real_data_loader.py                    # 三大真实公开数据集提取与 10 门店网络抽取流水线
│   └── evaluation.py                          # WAPE / MAE / Bias 业务场景切片（Overall/Normal/Cannib/Spill）
│
├── experiments/                               # 四大独立科学实验主程序
│   ├── exp1_linear_feature_ablation.py        # 实验一：线性特征与样本池化消融阶梯 (L0 -> L4)
│   ├── exp2_model_family_benchmark.py         # 实验二：跨模型家族全网大比拼 + 零耦合负对照 + 相图扫描
│   ├── exp3_store_regime_breakdown.py         # 实验三：10 店场景拆解 + 幽灵库存退化 + 10 倍故障注入与熔断
│   └── exp4_public_real_datasets.py           # 实验四：三大真实公开基准（FRN-50K / Favorita / M5）实测
│
├── scripts/                                   # 工具与一键运行入口
│   ├── download_real_benchmarks.py            # 真实公开数据集断点续传下载器 (HuggingFace 镜像免鉴权)
│   ├── run_all_experiments.py                 # 全套实验一键主执行总控程序 (约 17 秒跑通全部实验)
│   └── plot_all_figures.py                    # 解耦绘图主脚本：读取 results/plot_data/*.csv 生成中英高清图表
│
├── data/                                      # 数据底座
│   ├── retail_network_2023_2025.csv           # 仿真 10 店 × 1,096 天完整数据集 (10,960 行)
│   └── public_benchmarks/                     # 清洗提取后的三大基准标准宽表 (FRN-50K / Favorita / M5)
│       └── raw/                               # [已 gitignore] 下载的原始 parquet 与 csv 缓存
│
├── results/                                   # 实验全量数值产物（完全解耦持久化）
│   ├── metrics/                               # 14 张核心指标统计 CSV（精确到小数后三位）
│   ├── plot_data/                             # 9 张专供 Matplotlib/Seaborn 绘图的纯静态 CSV (plot_fig1..fig9)
│   └── predictions/                           # 测试集逐店逐日预测原始明细流水表
│
├── figures/                                   # 出版级中英双语信息图 (36 张 PNG + WebP)
└── archive/                                   # 历史遗留单文件脚本与旧版 Notebook 归档
```

---

## 三、快速开始 (Quickstart)

### 1. 环境安装

推荐使用 Python 3.10+ 与 [uv](https://docs.astral.sh/uv/) 工具链：

```bash
git clone https://github.com/cuicaihao/retail-gnn-demand-forecast.git
cd retail-gnn-demand-forecast

# 使用 uv 安装核心运行依赖
uv pip install -r requirements.txt
```

### 2. 自动下载三大真实公开数据

```bash
# 从免鉴权公共镜像下载 FreshRetailNet-50K、Favorita 与 Walmart M5
python scripts/download_real_benchmarks.py
```

### 3. 一键运行全量实验套件与图表生成

```bash
# 一键导出解耦绘图数据并运行全部实验 1 -> 4（整套流水线耗时仅约 17 秒）
python scripts/run_all_experiments.py

# 附带 --plot 标志可同时一键生成全部出版级中英文高清图表 (PNG & WebP)
python scripts/run_all_experiments.py --plot

# 或随时通过解耦绘图脚本从 results/plot_data/ 独立渲染图表
python scripts/plot_all_figures.py --lang both
```

### 4. 单独运行指定子实验

```bash
python experiments/exp1_linear_feature_ablation.py  # 运行实验一：线性特征与合账本消融
python experiments/exp2_model_family_benchmark.py   # 运行实验二：跨模型家族对比与负对照
python experiments/exp3_store_regime_breakdown.py   # 运行实验三：门店全景体检与故障注入
python experiments/exp4_public_real_datasets.py     # 运行实验四：三大真实工业公开基准
```

### 5. 启动交互式 Notebook

```bash
# 启动轻量自包含 Notebook 进行逐节验证
jupyter notebook retail_gnn_demand_forecast.ipynb
```

---

## 四、详细技术实验报告与业务故事

为保持项目主页的整洁凝练，全部深度的数学推导、数据生成规范、逐店透视细分表与零售门店业务故事已移至独立的技术报告文档：

👉 **[experimental-report.md](experimental-report.md)**

技术报告完整包含：

- **零、零售网络拓扑与四层数据生成过程（DGP）数学推导**；
- **一、实验一：线性特征消融阶梯（$L0 \dots L4$）详细成绩单与三则零售故事**；
- **二、实验二：跨模型家族全网大比拼（GBDT 特征重要度、GNN 连续外推机制与负对照）**；
- **三、实验三：10 家门店全景体检透视（表 3A/3B/3C、幽灵库存退化与 10 倍故障熔断止损）**；
- **四、实验四：三大真实公开数据集评测总表与全部 30 家门店逐店 WAPE 表现明细**；
- **五、真实世界的“三阶梯落地定律”与生产技术选型指南**。

---

## 五、数据产物持久化与绘图解耦设计

本工程贯彻**数据计算与图表绘制完全解耦**的设计理念：

- **统计指标表格**：[`results/metrics/`](results/metrics/) 下保存了 14 张高精度统计表，支持对每个百分比进行审计核对。
- **绘图解耦数据**：[`results/plot_data/`](results/plot_data/) 下保存了 `plot_fig1` 至 `plot_fig9` 的静态宽表，可直接被绘图脚本读取渲染。
- **预测流水明细**：[`results/predictions/synth_test_predictions_all_models.csv`](results/predictions/synth_test_predictions_all_models.csv) 保存了全量原始预测值。

---

## 六、历史版本归档说明 (Archive)

在 [`archive/`](archive/) 目录中保存了早期版本（v1/v2）的单体脚本（`retail_gnn_benchmark.py`、`generate_figures.py`）以及内嵌图片的旧版 Notebook（`retail_gnn_demand_forecast_v1.ipynb`），供历史追溯与版本比对。最新生产代码请统一使用 `src/` 与 `experiments/`。

---

## 七、引用与关联阅读

- **深度技术长文**：[《用图神经网络做零售需求预测：数学原理、受控实验与生产落地边界》](https://cuicaihao.github.io/zh/posts/2026-10-24-fsm-stochastic-gnn-retail-forecast/)
- **English Post**: [Retail Demand Forecasting with Graph Neural Networks: Mathematical Principles, Controlled Experiments, and Production Reality Check](https://cuicaihao.github.io/posts/2026-10-24-fsm-stochastic-gnn-retail-forecast/)
- **系列前序文章**：
  - [《用 Prophet 做零售需求预测：从模型原理到工程实践》](https://cuicaihao.github.io/zh/posts/2026-10-01-prophet-retail-demand-forecast/)
  - [《从一棵树到一片森林：树模型的数学原理、算法演进与工程实践》](https://cuicaihao.github.io/zh/posts/2026-10-17-from-a-single-tree-to-a-forest-mathematical-principles-algorithmic-evolution-and-engineering-practices-of-tree-models/)
