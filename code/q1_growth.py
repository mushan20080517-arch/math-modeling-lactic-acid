# -*- coding: utf-8 -*-
"""问题 1：菌体生长动力学建模（lg(CFU/mL) ~ t）"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import common as C
from growth import MODELS

t = C.T1_T
y = C.T1_L
n = len(t)

TGRID = np.linspace(0.0, 20.0, 4001)


def fit_model(spec, tt, yy):
    res = least_squares(lambda p: spec["fun"](p, tt) - yy, spec["p0"],
                        bounds=(spec["lb"], spec["ub"]), method="trf",
                        max_nfev=20000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    p = res.x
    yhat = spec["fun"](p, tt)
    st = C.stats(yy, yhat, spec["k"])
    J = res.jac
    dof = max(len(yy) - spec["k"], 1)
    s2 = st["rss"] / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(spec["k"], np.nan)
    return p, se, yhat, st


def derivatives(spec, p):
    """密网格数值微分，给出最大增长速率、出现时刻与 t→∞ 的极限。"""
    v = spec["fun"](p, TGRID)
    d = np.gradient(v, TGRID)
    i = int(np.argmax(d))
    Linf = float(spec["fun"](p, np.array([1e4]))[0])
    return float(d[i]), float(TGRID[i]), Linf, d


def main():
    results = {}
    print("=" * 78)
    print("问题1：菌体生长模型拟合与比较")
    print("=" * 78)
    for name, spec in MODELS.items():
        p, se, yhat, st = fit_model(spec, t, y)
        mumax, tstar, Linf, dgrid = derivatives(spec, p)
        loo_pred = np.zeros_like(y)
        for i in range(n):
            m = np.ones(n, bool)
            m[i] = False
            try:
                pi = least_squares(lambda q: spec["fun"](q, t[m]) - y[m], p,
                                   bounds=(spec["lb"], spec["ub"]), method="trf").x
                loo_pred[i] = spec["fun"](pi, t[i:i + 1])[0]
            except Exception:
                loo_pred[i] = np.nan
        press = float(np.nansum((y - loo_pred) ** 2))
        results[name] = dict(
            fun=spec["fun"].__name__, names=spec["names"], k=spec["k"],
            p=[float(v) for v in p], se=[float(v) for v in se],
            fit=yhat.tolist(), stats=st, r2adj=C.r2_adj(st["r2"], n, spec["k"]),
            mumax=mumax, t_mumax=tstar, Linf=Linf,
            L_at_14=float(spec["fun"](p, np.array([14.0]))[0]),
            press=press, rmse_loo=float(np.sqrt(press / n)),
        )
        print(f"\n{name}  (k={spec['k']})")
        for nm, val, s in zip(spec["names"], p, se):
            print(f"    {nm:6s} = {val:10.5f}  ± {s:.5f}")
        print(f"    R2={st['r2']:.5f}  RMSE={st['rmse']:.4f}  MAPE={st['mape']:.2f}%  "
              f"AIC={st['aic']:.2f}  BIC={st['bic']:.2f}  LOO-RMSE={results[name]['rmse_loo']:.4f}")
        print(f"    dL/dt_max = {mumax:.4f} lg/h  @ t={tstar:.2f} h ;  L_inf = {Linf:.4f}")

    best = min(results, key=lambda k: results[k]["stats"]["aic"])
    print(f"\n>>> 依 AIC 选出的模型：{best}")

    rng = np.random.default_rng(20261116)
    spec = MODELS[best]
    p_hat = np.array(results[best]["p"])
    yhat = np.array(results[best]["fit"])
    resid = y - yhat
    B = 600
    boot = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = yhat + resid[idx]
        try:
            rb = least_squares(lambda q: spec["fun"](q, t) - yb, p_hat,
                               bounds=(spec["lb"], spec["ub"]), method="trf").x
            mm, _, Li, _ = derivatives(spec, rb)
            boot.append([*rb, mm, Li])
        except Exception:
            continue
    boot = np.array(boot)
    q = np.percentile(boot, [2.5, 97.5], axis=0)
    results[best]["bootstrap"] = dict(
        B=len(boot),
        param_ci={nm: [float(q[0, i]), float(q[1, i])] for i, nm in enumerate(spec["names"])},
        mumax_ci=[float(q[0, -2]), float(q[1, -2])],
        Linf_ci=[float(q[0, -1]), float(q[1, -1])],
        mumax_samples=boot[:, -2].tolist(), Linf_samples=boot[:, -1].tolist(),
    )

    dydt_fd = np.gradient(y, t)
    from scipy.interpolate import UnivariateSpline
    spl = UnivariateSpline(t, y, s=n * 0.0045, k=3)
    dspl = spl.derivative()
    grid_fine = np.linspace(0, 14, 1401)
    ds = dspl(grid_fine)
    i = int(np.argmax(ds))
    results["_modelfree"] = dict(
        fd_rate=[float(v) for v in dydt_fd],
        spline_mumax=float(ds[i]), spline_t=float(grid_fine[i]),
        spline_curve=[[float(a), float(b)] for a, b in zip(grid_fine, ds)],
    )
    print(f"\n无模型对照：中心差分最大速率 {dydt_fd.max():.4f} lg/h；"
          f"平滑样条最大速率 {ds[i]:.4f} lg/h @ {grid_fine[i]:.2f} h")

    results["best"] = best
    C.json_dump(results, "q1_results.json")
    print("\n已保存 results/q1_results.json")


if __name__ == "__main__":
    main()
