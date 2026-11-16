# -*- coding: utf-8 -*-
"""Q1 候选生长模型（对数浓度空间 L = lg(CFU/mL)）。"""
from __future__ import annotations

import numpy as np


def f_malthus(p, tt):
    L0, mu = p
    return L0 + mu * tt


def f_malthus_lag(p, tt):
    L0, mu, lam = p
    return L0 + mu * np.maximum(tt - lam, 0.0)


def f_logistic(p, tt):
    L0, A, k, tm = p
    return L0 + A / (1.0 + np.exp(-k * (tt - tm)))


def f_gompertz(p, tt):
    L0, A, mue, lam = p
    e = np.e
    return L0 + A * np.exp(-np.exp(1.0 + mue * e / A * (lam - tt)))


def f_richards(p, tt):
    L0, A, k, tm, nu = p
    u = np.exp(-k * (tt - tm))
    return L0 + A * (1.0 + nu * u) ** (-1.0 / nu)


MODELS = {
    "M1 指数模型": dict(fun=f_malthus, p0=[7.6, 0.15], k=2,
                     lb=[6.5, 0.0], ub=[8.5, 2.0], names=["L0", "mu"]),
    "M2 含迟滞期指数模型": dict(fun=f_malthus_lag, p0=[7.6, 0.16, 2.0], k=3,
                        lb=[6.5, 0.0, -2.0], ub=[8.5, 2.0, 6.0], names=["L0", "mu", "lambda"]),
    "M3 Logistic 模型": dict(fun=f_logistic, p0=[7.6, 2.1, 0.5, 7.0], k=4,
                          lb=[6.5, 0.2, 0.01, 0.0], ub=[8.5, 6.0, 5.0, 30.0],
                          names=["L0", "A", "k", "tm"]),
    "M4 Gompertz 模型": dict(fun=f_gompertz, p0=[7.6, 2.1, 0.35, 2.5], k=4,
                          lb=[6.5, 0.2, 0.01, -3.0], ub=[8.5, 6.0, 3.0, 15.0],
                          names=["L0", "A", "mum", "lambda"]),
    "M5 Richards 模型": dict(fun=f_richards, p0=[7.6, 2.2, 0.5, 7.0, 0.6], k=5,
                          lb=[6.5, 0.2, 0.01, 0.0, 0.02], ub=[8.5, 6.0, 5.0, 30.0, 5.0],
                          names=["L0", "A", "k", "tm", "nu"]),
}
