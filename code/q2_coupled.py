# -*- coding: utf-8 -*-
"""问题 2："菌体生长—产酸—耗糖"联动模型

主体思想（Luedeking-Piret 耦合 + 产物抑制）:
    dP/dt = ( alpha * dL/dt + beta * X ) * (1 - P/Pm)
    dS/dt = -(1/Yps) * dP/dt - ms
其中 X = 10^(L-L0) 为相对菌体量，L(t) 由问题 1 的 Logistic 模型给出。
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

import common as C
from growth import f_logistic, f_gompertz, f_richards, f_malthus, f_malthus_lag

FUNCS = dict(f_logistic=f_logistic, f_gompertz=f_gompertz,
             f_richards=f_richards, f_malthus=f_malthus, f_malthus_lag=f_malthus_lag)

q1 = C.load_json("q1_results.json")
BEST = q1["best"]
LP = np.array(q1[BEST]["p"])
LFUN = FUNCS[q1[BEST]["fun"]]

t, L, P, S, SC = C.T1_T, C.T1_L, C.T1_P, C.T1_S, C.T1_SC
S0, n = C.S0_T1, len(C.T1_T)
L_LOW = LP[0]


def Lof(x):
    return float(LFUN(LP, np.array([float(x)]))[0])


def dLof(x):
    A, k, tm = LP[1], LP[2], LP[3]
    e = np.exp(-k * (float(x) - tm))
    return k * A * e / (1.0 + e) ** 2


SPEC = {
    "B3": dict(label="B3 无抑制耦合", k=4, p0=[2.4, 0.045, 1.03, 0.02],
               lb=[0.0, 0.0, 0.60, 0.0], ub=[20.0, 0.30, 1.50, 0.20]),
    "B4": dict(label="B4 耦合+产物抑制", k=5, p0=[2.4, 0.045, 11.0, 1.03, 0.03],
               lb=[0.0, 0.0, 6.0, 0.60, 0.0], ub=[20.0, 0.30, 400.0, 1.50, 0.20]),
    "B5": dict(label="B5 仅非生长偶联+抑制", k=4, p0=[0.10, 11.0, 1.03, 0.03],
               lb=[0.0, 6.0, 0.60, 0.0], ub=[0.30, 400.0, 1.50, 0.20]),
}


def simulate(theta, tt, form="B4", tmax=None):
    if form == "B4":
        alpha, beta, Pm, Yps, ms = theta
        inhib = True
    elif form == "B3":
        alpha, beta, Yps, ms = theta
        Pm = np.inf
        inhib = False
    else:                       # B5
        beta, Pm, Yps, ms = theta
        alpha = 0.0
        inhib = True
    tmax = tt[-1] if tmax is None else tmax

    def rhs(_t, y):
        p, _s = y
        f = max(1.0 - p / Pm, 0.0) if inhib else 1.0
        Xi = 10.0 ** (Lof(_t) - L_LOW)
        dP = max(alpha * dLof(_t) + beta * Xi, 0.0) * f
        return [dP, -dP / Yps - ms]

    sol = solve_ivp(rhs, (0.0, tmax), [0.0, S0], t_eval=tt, method="LSODA",
                    rtol=1e-10, atol=1e-12, max_step=0.05)
    return sol.y


def fit(form, tt=None, nstart=8):
    tt = t if tt is None else tt
    sp = SPEC[form]
    if form in ("B1", "B2"):
        return None
    p0, lb, ub = np.array(sp["p0"]), np.array(sp["lb"]), np.array(sp["ub"])

    def resid(q):
        y = simulate(q, tt, form)
        return np.concatenate([(y[0] - P[:len(tt)]) / C.SIGMA_P,
                               (y[1] - S[:len(tt)]) / C.SIGMA_S])

    best = None
    for s in range(nstart):
        rng = np.random.default_rng(100 + s)
        q0 = np.clip(p0 * (1 + 0.25 * rng.normal(size=len(p0))), lb + 1e-8, ub - 1e-8)
        try:
            r = least_squares(resid, q0, bounds=(lb, ub), method="trf",
                              x_scale="jac", max_nfev=6000,
                              xtol=1e-13, ftol=1e-13, gtol=1e-13)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


print("=" * 78)
print("问题2：产酸—耗糖—生长联动模型")
print("=" * 78)

# ---------------- B1 化学计量（产率）模型 ----------------
Yps_yield = float(np.sum(SC * P) / np.sum(SC ** 2))
P_yield = Yps_yield * SC
st_yield = C.stats(P, P_yield, 1)
A1 = np.column_stack([np.ones(n), SC])
coef_int, *_ = np.linalg.lstsq(A1, P, rcond=None)
st_int = C.stats(P, A1 @ coef_int, 2)
r_ps = float(np.corrcoef(P, SC)[0, 1])
print(f"[B1 产率模型] Yps={Yps_yield:.4f} g/g  R2={st_yield['r2']:.5f} "
      f"RMSE={st_yield['rmse']:.4f}  r(P,Sc)={r_ps:.5f}")
print(f"    含截距: P={coef_int[1]:.4f}*Sc{coef_int[0]:+.4f}   R2={st_int['r2']:.5f}")

# ---------------- B2 静态 Luedeking-Piret ----------------
A2 = np.column_stack([L - L[0], t])
coef_lp, *_ = np.linalg.lstsq(A2, P, rcond=None)
P_lp = A2 @ coef_lp
st_lp = C.stats(P, P_lp, 2)
print(f"[B2 L-P静态] alpha={coef_lp[0]:.4f}  beta={coef_lp[1]:.4f}  "
      f"R2={st_lp['r2']:.5f}  RMSE={st_lp['rmse']:.4f}")

# ---------------- B3~B5 动力学耦合 ----------------
fits, summary = {}, {}
for code in ["B3", "B4", "B5"]:
    form = code
    r = fit(form)
    fits[form] = r
    y = simulate(r.x, t, form)
    sP, sS = C.stats(P, y[0], len(r.x)), C.stats(S, y[1], len(r.x))
    rss_tot = float(np.sum(r.fun ** 2))
    summary[code] = dict(label=SPEC[code]["label"], params=[float(v) for v in r.x],
                         P=y[0].tolist(), S=y[1].tolist(),
                         statsP=sP, statsS=sS, k=len(r.x), rss=rss_tot,
                         aic=float(len(r.fun) * np.log(rss_tot / len(r.fun)) + 2 * len(r.x)))
    print(f"\n[{SPEC[code]['label']}] k={len(r.x)}  参数={np.round(r.x, 5)}")
    print(f"    P: R2={sP['r2']:.5f} RMSE={sP['rmse']:.4f} MAPE={sP['mape']:.2f}%   "
          f"S: R2={sS['r2']:.5f} RMSE={sS['rmse']:.4f}")
    print(f"    P_fit = {np.round(y[0], 2)}")

BESTM = "B4"
fitB = fits[BESTM]
alpha, beta, Pm, Yps, ms = fitB.x
yB = simulate(fitB.x, t, "B4")
Ph0, Sh0 = yB

# ---------------- 参数不确定度 ----------------
dof = max(2 * n - len(fitB.x), 1)
s2 = float(np.sum(fitB.fun ** 2)) / dof
try:
    cov = s2 * np.linalg.pinv(fitB.jac.T @ fitB.jac)
    se = np.sqrt(np.abs(np.diag(cov)))
except Exception:
    se = np.full(len(fitB.x), np.nan)

rng = np.random.default_rng(2468)
rp, rs = P - Ph0, S - Sh0
boot = []
for b in range(400):
    Pb = Ph0 + rp[rng.integers(0, n, n)]
    Sb = Sh0 + rs[rng.integers(0, n, n)]

    def rb(q, Pb=Pb, Sb=Sb):
        a, c = simulate(q, t, "B4")
        return np.concatenate([(a - Pb) / C.SIGMA_P, (c - Sb) / C.SIGMA_S])


    try:
        boot.append(least_squares(rb, fitB.x,
                                  bounds=(SPEC["B4"]["lb"], SPEC["B4"]["ub"]),
                                  method="trf", max_nfev=2500).x)
    except Exception:
        pass
boot = np.array(boot)
ci = np.percentile(boot, [2.5, 97.5], axis=0) if len(boot) > 20 else np.full((2, 5), np.nan)
names = ["alpha", "beta", "Pm", "Yps", "ms"]
print(f"\n[自助法 95% 置信区间] B={len(boot)}")
for i, nm in enumerate(names):
    print(f"   {nm:6s} {fitB.x[i]:10.5f}  se={se[i]:8.5f}  95%CI [{ci[0, i]:.5f}, {ci[1, i]:.5f}]")

# ---------------- 产酸来源分解 ----------------
tt = np.linspace(0, t[-1], 2801)
PV = np.interp(tt, t, Ph0)
P_g, P_m = 0.0, 0.0
for i in range(len(tt) - 1):
    dt = tt[i + 1] - tt[i]
    x = tt[i]
    f = max(1.0 - PV[i] / Pm, 0.0)
    P_g += max(alpha * dLof(x), 0.0) * f * dt
    P_m += beta * 10.0 ** (Lof(x) - L_LOW) * f * dt
share_g = P_g / (P_g + P_m)
print(f"\n[产酸来源] 生长偶联项 {P_g:.3f} g/L ({share_g*100:.1f}%)；"
      f"非生长偶联项 {P_m:.3f} g/L ({(1-share_g)*100:.1f}%)")

# ---------------- 逐点拟合表 ----------------
table = []
for i in range(n):
    table.append(dict(t=float(t[i]), P_obs=float(P[i]), P_fit=float(Ph0[i]),
                      P_err=float(Ph0[i] - P[i]),
                      P_rel=float((Ph0[i] - P[i]) / P[i] * 100) if P[i] else 0.0,
                      S_obs=float(S[i]), S_fit=float(Sh0[i]),
                      S_err=float(Sh0[i] - S[i]),
                      ratio=float(P[i] / SC[i]) if SC[i] else None))
print("\n[逐点拟合]   t   P_obs  P_fit   err   rel%   S_obs  S_fit   err   P/Sc")
for r in table:
    print(f"          {r['t']:4.1f} {r['P_obs']:6.2f} {r['P_fit']:6.2f} {r['P_err']:+6.2f} "
          f"{r['P_rel']:+6.2f} {r['S_obs']:7.2f} {r['S_fit']:6.2f} {r['S_err']:+6.2f}  "
          f"{(r['ratio'] or float('nan')):5.3f}")

# ---------------- 外推预测检验 ----------------
fit6 = fit("B4", t[:6])
y6 = simulate(fit6.x, t[:8], "B4")
print("\n[预测检验] 用 0-10 h 六个点建模，外推 12 h、14 h：")
print(f"   拟合参数 {np.round(fit6.x, 4)}")
print(f"   P(12) 预测 {y6[0][6]:.3f} (实测 {P[6]:.2f})；P(14) 预测 {y6[0][7]:.3f} (实测 {P[7]:.2f})")

# ---------------- 敏感性 ----------------
sens = {}
base = float(Ph0[-1])
for i, nm in enumerate(names):
    q = fitB.x.copy()
    q[i] *= 1.1
    try:
        sens[nm] = float((simulate(q, t, "B4")[0][-1] - base) / base * 100)
    except Exception:
        sens[nm] = float("nan")
print("\n[敏感性] 参数 +10% 对终点乳酸的影响 (%):",
      {k: round(v, 3) for k, v in sens.items()})

out = dict(
    growth_basis=dict(best=BEST, p=LP.tolist(), L0=float(L_LOW)),
    yield_model=dict(Yps=Yps_yield, stats=st_yield, P=P_yield.tolist(), r=r_ps,
                     intercept=float(coef_int[0]), slope=float(coef_int[1]),
                     stats_int=st_int),
    lp_static=dict(alpha=float(coef_lp[0]), beta=float(coef_lp[1]), stats=st_lp),
    models={k: dict(label=v["label"], params=v["params"], statsP=v["statsP"],
                    statsS=v["statsS"], k=v["k"], aic=v["aic"], rss=v["rss"],
                    P=v["P"], S=v["S"])
            for k, v in summary.items()},
    best_model=summary[BESTM]["label"],
    best=dict(params=dict(zip(names, [float(v) for v in fitB.x])),
              se=dict(zip(names, [float(v) for v in se])),
              ci={nm: [float(ci[0, i]), float(ci[1, i])] for i, nm in enumerate(names)},
              statsP=summary[BESTM]["statsP"], statsS=summary[BESTM]["statsS"],
              P=Ph0.tolist(), S=Sh0.tolist()),
    decomposition=dict(P_growth=P_g, P_maint=P_m, share_growth=share_g),
    table=table,
    predict=dict(p12=float(y6[0][6]), p14=float(y6[0][7]), fit6=[float(v) for v in fit6.x]),
    sensitivity=sens,
    dPdt=np.gradient(P, t).tolist(), dLdt=np.gradient(L, t).tolist(),
    X_data=(10.0 ** (L - L_LOW)).tolist(),
)
C.json_dump(out, "q2_results.json")
print("\n已保存 results/q2_results.json")
