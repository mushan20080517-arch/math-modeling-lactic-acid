# -*- coding: utf-8 -*-
"""公共模块：数据、评价指标、绘图风格。

2026 新手数学建模竞赛 A 题  乳酸发酵工艺的规律与优化
"""
from __future__ import annotations

import json
import os

import numpy as np

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# --------------------------------------------------------------------------
# 试验一：标准配方（初糖 35 g/L，37 ℃，L-1 菌株）分批发酵过程数据
#   t  时间/h
#   L  菌体浓度 lg(CFU/mL)
#   P  乳酸浓度 g/L
#   Sc 累计葡萄糖消耗 g/L
#   S  残糖 g/L = S0 - Sc
# --------------------------------------------------------------------------
S0_T1 = 35.0
T1_T = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
T1_L = np.array([7.63, 7.75, 8.15, 8.79, 9.38, 9.59, 9.66, 9.68])
T1_P = np.array([0.00, 0.36, 1.08, 2.88, 6.12, 7.83, 9.09, 9.99])
T1_SC = np.array([0.0, 0.4, 1.2, 3.2, 6.3, 7.9, 9.0, 10.1])
T1_S = S0_T1 - T1_SC

# --------------------------------------------------------------------------
# 试验二：不同初糖浓度对比（L-2 菌株，40 ℃，pH 9.0 恒定）
#   S0 初糖 g/L, te 终点时间 h, OD 终点 OD600, P 终点乳酸 g/L,
#   Sr 终点残糖 g/L, ravg 平均产酸速率 g/(L·h), rmax 最大产酸速率 g/(L·h)
# --------------------------------------------------------------------------
T2_S0 = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 150.0])
T2_TE = np.array([9.0, 18.0, 33.0, 63.0, 90.0, 108.0])
T2_OD = np.array([1.99, 4.19, 6.25, 7.69, 9.19, 9.72])
T2_P = np.array([19.6, 37.3, 57.9, 77.4, 96.0, 109.9])
T2_SR = np.array([0.1, 0.0, 0.0, 0.5, 0.0, 35.7])
T2_RAVG = np.array([2.18, 2.07, 1.75, 1.23, 1.07, 1.02])
T2_RMAX = np.array([3.20, 3.30, 3.40, 3.10, 3.30, 2.60])

# 假定的测量误差（用于联合加权最小二乘，权重 = 1/sigma）
SIGMA_L = 0.05      # lg(CFU/mL)
SIGMA_P = 0.15      # g/L
SIGMA_S = 0.15      # g/L


# --------------------------------------------------------------------------
# 评价指标
# --------------------------------------------------------------------------
def stats(y, yhat, k):
    """拟合优度与信息准则。k = 参数个数（含误差方差之外的自由参数）。"""
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    rmse = float(np.sqrt(rss / n))
    mae = float(np.mean(np.abs(y - yhat)))
    mask = np.abs(y) > 1e-12
    mape = float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100) if mask.any() else float("nan")
    rss_safe = max(rss, 1e-12)
    aic = n * np.log(rss_safe / n) + 2 * k
    bic = n * np.log(rss_safe / n) + k * np.log(n)
    return dict(n=n, k=k, rss=rss, r2=r2, rmse=rmse, mae=mae, mape=mape,
                aic=float(aic), bic=float(bic))


def r2_adj(r2, n, k):
    return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)


def json_dump(obj, name):
    path = os.path.join(RES, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    return path


def load_json(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# 绘图风格
# --------------------------------------------------------------------------
C_BLUE = "#2b6cb0"
C_RED = "#c53030"
C_GREEN = "#2f855a"
C_ORANGE = "#dd6b20"
C_PURPLE = "#6b46c1"
C_GRAY = "#718096"
COLORS = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE]


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })
    return plt


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path
