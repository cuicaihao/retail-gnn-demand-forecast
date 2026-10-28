# Archive / 历史遗留文件归档

本目录保存了项目早期（v1/v2 版本）的单文件单体实验与绘图脚本，供历史对比与参考查阅：

1. `retail_gnn_benchmark.py`:
   早期的单文件脚本（约 690 行），已由现代模块化架构全面替代：
   - 数据生成与拓扑逻辑已解耦迁移至 `src/data_generator.py`；
   - 特征矩阵构造迁移至 `src/feature_builders.py`；
   - 线性与 Tobit 求解器迁移至 `src/models_linear.py`；
   - GBDT 树模型与两层端到端 Hetero-GNN 迁移至 `src/models_tree.py` 与 `src/models_gnn.py`；
   - 四项独立科学实验迁移至 `experiments/` 目录；
   - 一键执行总控入口统一为 `scripts/run_all_experiments.py`。

2. `generate_figures.py`:
   基于早期单体脚本输出内存字典的绘图脚本。当前最新架构下，所有绘图数据均已 100% 静态解耦持久化至 `results/plot_data/`，支持完全独立的图表绘制。

3. `retail_gnn_demand_forecast_v1.ipynb`:
   早期内嵌庞大 base64 图片（约 4MB）的单体 Notebook，已被根目录全新的轻量自包含版本（29KB）替代。

