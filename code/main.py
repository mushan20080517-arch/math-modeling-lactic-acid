# -*- coding: utf-8 -*-
"""一键复现：依次运行四个问题的建模脚本与绘图脚本。

用法:  python main.py
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    ("问题 1  菌体生长模型", "q1_growth.py"),
    ("问题 2  生长-产酸-耗糖耦合", "q2_coupled.py"),
    ("问题 3  初糖优化", "q3_opt.py"),
    ("问题 3  稳健性检验", "q3_robust.py"),
    ("问题 4  过程强化", "q4_strategy.py"),
    ("绘图", "make_figures.py"),
]


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for label, script in SCRIPTS:
        print("\n" + "=" * 70)
        print(f">>> {label}  ({script})")
        print("=" * 70)
        r = subprocess.run([sys.executable, script], cwd=HERE, env=env)
        if r.returncode != 0:
            print(f"!!! {script} 运行失败，退出码 {r.returncode}")
            return r.returncode
    print("\n全部完成。结果见 results/，图见 figures/。")


if __name__ == "__main__":
    sys.exit(main())
