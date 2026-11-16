# -*- coding: utf-8 -*-
"""问题 3 的稳健性检验：留一重拟合 + 纯数据插值优化。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import common as C

S0v, TE, PE, RMAX = C.T2_S0, C.T2_TE, C.T2_P, C.T2_RMAX
n = len(S0v)


def rate(theta, S0x, Pp):
    Vmax, Ks, Ki, Pcap, Y = theta[:5]
    Ss = S0x - Pp / Y
    if Ss <= 0:
        return 0.0
    return Vmax * Ss / (Ks + Ss + Ss * Ss / Ki) * max(1.0 - Pp / Pcap, 0.0)


def end_point(theta, S0x):
    vstop = theta[5]
    hi = min(theta[4] * S0x, theta[3]) * (1 - 1e-10)
    lo = 1e-10
    f = lambda P: rate(theta, S0x, P) - vstop
    if f(lo) <= 0:
        return 0.0, 0.0
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if f(m) > 0:
            lo = m
        else:
            hi = m
    Pe = 0.5 * (lo + hi)
    g = np.linspace(0.0, Pe, 401)
    rr = np.maximum([rate(theta, S0x, x) for x in g], 1e-9)
    return float(np.trapezoid(1 / rr, g)), float(Pe)


def predict(theta, S0x):
    te, Pe = end_point(theta, S0x)
    g = np.linspace(0.0, Pe, 201)
    return te, Pe, max(rate(theta, S0x, x) for x in g)


def fit(idx):
    p0 = [3.65, 2.28, 513.0, 111.5, 0.983, 0.124]
    lb = [0.5, 0.05, 5.0, 80.0, 0.6, 0.005]
    ub = [20.0, 200.0, 5000.0, 160.0, 1.15, 0.5]

    def resid(th):
        out = []
        for i in idx:
            te, Pe, rm = predict(th, S0v[i])
            out += [(Pe - PE[i]) / 2.0, (te / TE[i] - 1) / 0.08,
                    (rm / RMAX[i] - 1) / 0.08]
        return np.array(out)

    best = None
    for s in range(4):
        rng = np.random.default_rng(s)
        q0 = np.clip(np.array(p0) * (1 + 0.15 * rng.normal(size=6)), np.array(lb) + 1e-6,
                     np.array(ub) - 1e-6)
        try:
            r = least_squares(resid, q0, bounds=(lb, ub), method="trf", max_nfev=6000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


COST = dict(p_P=0.0100, c_var=0.0015, c_vol=0.120, p_S=0.0032, c_op=0.0015, t_turn=6.0)


def trajectory(theta, S0x, dt=0.25):
    te, Pe = end_point(theta, S0x)
    tt = np.arange(0.0, max(te, 1e-6) + 1e-9, dt)
    if len(tt) < 3:
        tt = np.array([0.0, max(te, 1e-6)])
    Pv, p = [], 0.0
    for _ in tt:
        Pv.append(p)
        p += rate(theta, S0x, p) * dt
    return tt, np.minimum(np.array(Pv), Pe)


def best_S0(theta):
    g = np.arange(20.0, 150.51, 1.0)
    Js, Cs = [], []
    for s in g:
        tt, Pv = trajectory(theta, s)
        Jb = -1e18
        for th in np.arange(4.0, tt[-1] + 1e-9, 1.0):
            Pt = float(np.interp(th, tt, Pv))
            J = (COST["p_P"] * Pt - COST["p_S"] * s - COST["c_var"] * Pt
                 - COST["c_vol"] - COST["c_op"] * th) / (th + COST["t_turn"])
            Jb = max(Jb, J)
        Js.append(Jb)
        Cs.append((COST["p_S"] * s + COST["c_op"] * tt[-1] + COST["c_var"] * Pv[-1]
                   + COST["c_vol"]) / Pv[-1] * 1000)
    Js, Cs = np.array(Js), np.array(Cs)
    return float(g[int(np.argmax(Js))]), float(g[int(np.argmin(Cs))]), Js


theta_all = np.array([3.656651706626188, 2.2829330001314436, 513.5953479204624,
                      111.54699156917003, 0.9830489594689465, 0.12390440709427472])
g = np.arange(20.0, 150.51, 1.0)
sJ0, sC0, Js0 = best_S0(theta_all)
print(f"[全样本] 最优 S0(J)={sJ0:.0f}, S0(C)={sC0:.0f}")

rows = []
for i in range(n):
    idx = [j for j in range(n) if j != i]
    th = fit(idx)
    sJ, sC, _ = best_S0(th)
    rows.append(dict(drop=float(S0v[i]), S0_J=sJ, S0_C=sC,
                     params=[float(v) for v in th]))
    print(f"   去掉 {S0v[i]:5.0f}: S0(J)={sJ:.0f}  S0(C)={sC:.0f}")

sJ = [r["S0_J"] for r in rows]
sC = [r["S0_C"] for r in rows]
print(f"   LOO 范围: J 最优 {min(sJ):.0f}~{max(sJ):.0f}；成本最优 {min(sC):.0f}~{max(sC):.0f}")

# 纯数据插值优化（不使用模型）
g2 = np.arange(20.0, 130.0, 0.5)
Pd = np.interp(g2, S0v, PE)
Td = np.interp(g2, S0v, TE)
Jd = []
for s, P_, T_ in zip(g2, Pd, Td):
    ts = np.linspace(4.0, max(T_, 5.0), 120)
    Ps = np.interp(ts, [0, T_], [0, P_])        # 线性近似 P(t)
    J = (COST["p_P"] * Ps - COST["p_S"] * s - COST["c_var"] * Ps
         - COST["c_vol"] - COST["c_op"] * ts) / (ts + COST["t_turn"])
    Jd.append(float(np.max(J)))
Jd = np.array(Jd)
k = int(np.argmax(Jd))
print(f"[纯数据插值] 最优 S0={g2[k]:.1f} g/L, J={Jd[k]:.5f}")

# J 平台宽度
m = (g >= 70) & (g <= 95)
span = (Js0[m].max() - Js0[m].min()) / Js0[m].max() * 100
print(f"[平台] 70~95 g/L 内 J 变化 {span:.2f}%")

C.json_dump(dict(loo=rows, loo_range_J=[min(sJ), max(sJ)], loo_range_C=[min(sC), max(sC)],
                 full=dict(S0_J=sJ0, S0_C=sC0),
                 data_interp=dict(S0=float(g2[k]), J=float(Jd[k])),
                 plateau_pct=float(span), grid=g.tolist(), J=Js0.tolist()),
            "q3_robust.json")
print("saved results/q3_robust.json")
