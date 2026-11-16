# -*- coding: utf-8 -*-
"""问题 3：初糖浓度的影响规律、终点模型与工艺优化"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, least_squares

import common as C

S0v, TE, OD, PE, SR, RAVG, RMAX = (C.T2_S0, C.T2_TE, C.T2_OD, C.T2_P,
                                   C.T2_SR, C.T2_RAVG, C.T2_RMAX)
n = len(S0v)

print("=" * 78)
print("问题3：初糖浓度的影响与工艺优化")
print("=" * 78)

# --------------------------------------------------------------------------
# 0. 描述统计
# --------------------------------------------------------------------------
tab = []
for i in range(n):
    tab.append(dict(S0=float(S0v[i]), te=float(TE[i]), OD=float(OD[i]), P=float(PE[i]),
                    Sr=float(SR[i]), ravg=float(RAVG[i]), rmax=float(RMAX[i]),
                    Y_gross=float(PE[i] / S0v[i]),
                    Y_net=float(PE[i] / (S0v[i] - SR[i])),
                    rho=float(PE[i] / TE[i]),
                    OD_yield=float(OD[i] / (S0v[i] - SR[i]))))
print(" S0    te     OD     P     Sr   P/S0  P/(S0-Sr)  P/te   OD/(S0-Sr)")
for r in tab:
    print(f"{r['S0']:3.0f} {r['te']:5.0f} {r['OD']:6.2f} {r['P']:6.1f} {r['Sr']:6.1f} "
          f"{r['Y_gross']:6.3f} {r['Y_net']:9.3f} {r['rho']:6.3f} {r['OD_yield']:8.3f}")


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


print(f"\n相关系数: r(S0,P)={corr(S0v,PE):.3f}  r(S0,r_avg)={corr(S0v,RAVG):.3f}  "
      f"r(S0,te)={corr(S0v,TE):.3f}  r(S0,r_max)={corr(S0v,RMAX):.3f}")

# --------------------------------------------------------------------------
# 1. 终点产量模型
# --------------------------------------------------------------------------
def m_linear(p, s):
    return p[0] * s


def m_mm(p, s):
    return p[0] * s / (p[1] + s)


def m_cap(p, s):
    Y, Pcap = p
    return np.minimum(Y * s, Pcap)


ends = {}
for name, model, p0, lb, ub, k in [
        ("C1 线性 P=a·S0", m_linear, [0.9], [0.5], [1.5], 1),
        ("C2 Michaelis-Menten", m_mm, [100.0, 60.0], [1.0, 1.0], [1e4, 1e4], 2),
        ("C3 产率+抑制上限", m_cap, [0.96, 108.0], [0.5, 80.0], [1.2, 160.0], 2)]:
    r = least_squares(lambda p: model(p, S0v) - PE, p0, bounds=(lb, ub),
                      method="trf", max_nfev=20000)
    st = C.stats(PE, model(r.x, S0v), k)
    ends[name] = dict(params=[float(v) for v in r.x], stats=st,
                      fit=model(r.x, S0v).tolist())
    print(f"\n[{name}] 参数={np.round(r.x, 4)}  R2={st['r2']:.5f} RMSE={st['rmse']:.4f} "
          f"AIC={st['aic']:.2f}")

Y_end, Pcap_end = ends["C3 产率+抑制上限"]["params"]
S0crit = Pcap_end / Y_end
print(f"\n>>> 产率-上限模型：Y={Y_end:.4f} g/g, Pcap={Pcap_end:.2f} g/L, "
      f"临界初糖 S0*={S0crit:.1f} g/L")
print(f"    150 g/L 残糖预测 {150 - Pcap_end/Y_end:.2f} g/L（实测 35.70）")

# OD600 与消耗糖的关系
A = np.column_stack([np.ones(n), S0v - SR])
c_od, *_ = np.linalg.lstsq(A, OD, rcond=None)
st_od = C.stats(OD, A @ c_od, 2)
print(f"\n[菌体量] OD600 = {c_od[1]:.4f}·(S0-Sr){c_od[0]:+.4f}, R2={st_od['r2']:.4f}")

# --------------------------------------------------------------------------
# 2. 动力学模型  dP/dt = Vmax·S/(Ks+S+S²/Ki)·(1-P/Pcap),  S=S0-P/Y
#    终点判据：产酸速率降至 vstop（对应"残糖基本不再下降"）
# --------------------------------------------------------------------------
def rate(theta, S0x, Pp):
    Vmax, Ks, Ki, Pcap, Y = theta[:5]
    Ss = S0x - Pp / Y
    if Ss <= 0:
        return 0.0
    return Vmax * Ss / (Ks + Ss + Ss * Ss / Ki) * max(1.0 - Pp / Pcap, 0.0)


def end_point(theta, S0x):
    Vmax, Ks, Ki, Pcap, Y, vstop = theta
    hi = min(Y * S0x, Pcap) * (1 - 1e-10)
    f = lambda P: rate(theta, S0x, P) - vstop
    if f(1e-12) <= 0:
        return 0.0, 0.0
    lo = 1e-10
    for _ in range(60):                     # 二分（避免 brentq 在端点失败）
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    Pe = 0.5 * (lo + hi)
    g = np.linspace(0.0, Pe, 801)
    rr = np.array([rate(theta, S0x, x) for x in g])
    rr = np.maximum(rr, 1e-9)
    te = float(np.trapezoid(1.0 / rr, g))
    return te, float(Pe)


def predict(theta, S0x):
    te, Pe = end_point(theta, S0x)
    g = np.linspace(0.0, Pe, 801)
    return dict(te=te, P=Pe, rmax=float(max(rate(theta, S0x, x) for x in g)),
                ravg=Pe / te if te > 0 else 0.0,
                Sr=float(S0x - Pe / theta[4]))


def resid(theta, wP=2.0, wrel=0.08):
    """相对误差加权：三个指标量纲不同，用相对残差更合理。"""
    out = []
    for i in range(n):
        q = predict(theta, S0v[i])
        out += [(q["P"] - PE[i]) / wP, (q["te"] / TE[i] - 1) / wrel,
                (q["rmax"] / RMAX[i] - 1) / wrel]
    return np.array(out)


p0 = [4.5, 5.0, 250.0, 110.0, 0.96, 0.05]
lb = [0.5, 0.05, 5.0, 80.0, 0.6, 0.005]
ub = [20.0, 200.0, 5000.0, 160.0, 1.15, 0.5]
best = None
for s in range(12):
    rng = np.random.default_rng(11 + s)
    q0 = np.clip(np.array(p0) * (1 + 0.2 * rng.normal(size=6)), np.array(lb) + 1e-6,
                 np.array(ub) - 1e-6)
    try:
        r = least_squares(resid, q0, bounds=(lb, ub), method="trf", x_scale="jac",
                          max_nfev=20000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
    except Exception:
        continue
    if best is None or r.cost < best.cost:
        best = r
Vmax, Ks, Ki, Pcap, Y, vstop = best.x
print(f"\n[Q3 动力学模型] Vmax={Vmax:.4f}  Ks={Ks:.4f}  Ki={Ki:.2f}  "
      f"Pcap={Pcap:.3f}  Y={Y:.4f}  vstop={vstop:.4f}")
print(" S0   te_pred te_obs  P_pred  P_obs  rmax_pred rmax_obs  ravg_pred ravg_obs  Sr_pred")
pred_tab = []
for i in range(n):
    q = predict(best.x, S0v[i])
    pred_tab.append(dict(S0=float(S0v[i]), **{k: float(v) for k, v in q.items()},
                         te_obs=float(TE[i]), P_obs=float(PE[i]), rmax_obs=float(RMAX[i]),
                         ravg_obs=float(RAVG[i]), Sr_obs=float(SR[i])))
    print(f"{S0v[i]:3.0f} {q['te']:8.1f} {TE[i]:6.0f} {q['P']:7.2f} {PE[i]:6.1f} "
          f"{q['rmax']:9.2f} {RMAX[i]:8.2f} {q['ravg']:9.3f} {RAVG[i]:8.3f} {q['Sr']:7.2f}")

r2_t = C.stats(TE, [p["te"] for p in pred_tab], 6)
r2_P = C.stats(PE, [p["P"] for p in pred_tab], 6)
r2_r = C.stats(RMAX, [p["rmax"] for p in pred_tab], 6)
r2_ra = C.stats(RAVG, [p["ravg"] for p in pred_tab], 6)
print(f"\n   R2: te={r2_t['r2']:.4f}  P={r2_P['r2']:.4f}  rmax={r2_r['r2']:.4f}  "
      f"r_avg={r2_ra['r2']:.4f}")

# --- 单独拟合最大产酸速率：Haldane 底物抑制 ---
def rmax_haldane(p, s):
    Vm, Ks2, Ki2 = p
    return Vm * s / (Ks2 + s + s * s / Ki2)


r_rm = least_squares(lambda p: rmax_haldane(p, S0v) - RMAX, [3.6, 3.0, 300.0],
                     bounds=([0.5, 0.01, 5.0], [20.0, 200.0, 1e5]), method="trf")
st_rm = C.stats(RMAX, rmax_haldane(r_rm.x, S0v), 3)
print(f"\n[最大产酸速率] Haldane: Vmax={r_rm.x[0]:.4f} Ks={r_rm.x[1]:.4f} "
      f"Ki={r_rm.x[2]:.1f}  R2={st_rm['r2']:.4f} RMSE={st_rm['rmse']:.4f}")
print(f"   拟合值 {np.round(rmax_haldane(r_rm.x, S0v), 2)}")

dof = max(len(best.fun) - 6, 1)
s2 = float(np.sum(best.fun ** 2)) / dof
cov = s2 * np.linalg.pinv(best.jac.T @ best.jac)
se = np.sqrt(np.abs(np.diag(cov)))
nm6 = ["Vmax", "Ks", "Ki", "Pcap", "Y", "vstop"]
for nm, v, s_ in zip(nm6, best.x, se):
    print(f"   {nm:6s} = {v:10.4f} ± {s_:.4f}")

# --------------------------------------------------------------------------
# 3. 工艺优化
# --------------------------------------------------------------------------
COST = dict(p_P=0.0100, c_var=0.0015, c_vol=0.120,
            p_S=0.0032, c_op=0.0015, t_turn=6.0)


def trajectory(theta, S0x, dt=0.25):
    te, Pe = end_point(theta, S0x)
    tt = np.arange(0.0, te + 1e-9, dt)
    if len(tt) < 3:
        tt = np.array([0.0, te])
    Pv = []
    p = 0.0
    for x in tt:
        Pv.append(p)
        p += rate(theta, S0x, p) * dt
    Pv = np.array(Pv)
    Pv = np.minimum(Pv, Pe)
    return tt, Pv, te, Pe


def economics(S0x, th, cost=COST, tt=None, Pv=None):
    if tt is None:
        tt, Pv, te, Pe = trajectory(best.x, S0x)
    Pt = float(np.interp(th, tt, Pv))
    num = (cost["p_P"] * Pt - cost["p_S"] * S0x - cost["c_var"] * Pt
           - cost["c_vol"] - cost["c_op"] * th)
    return num / (th + cost["t_turn"]), Pt


grid_S0 = np.arange(20.0, 150.01, 1.0)
prof = []
for s in grid_S0:
    tt, Pv, te, Pe = trajectory(best.x, s)
    rho = Pv / (tt + COST["t_turn"])
    i = int(np.argmax(rho))
    Jb, Pb, tb = -1e18, 0.0, 0.0
    for th in np.arange(2.0, te + 1e-9, 0.5):
        J, Pt = economics(s, th, tt=tt, Pv=Pv)
        if J > Jb:
            Jb, Pb, tb = J, Pt, th
    prof.append(dict(S0=float(s), te=float(te), P=float(Pe),
                     rho=float(Pe / (te + COST["t_turn"])),
                     t_star=float(tt[i]), rho_star=float(rho[i]),
                     Sr=float(s - Pe / Y), Yf=float(Pe / s),
                     J=float(Jb), t_harvest=float(tb), P_harvest=float(Pb)))
prof = {k: np.array([p[k] for p in prof]) for k in prof[0]}

iJ = int(np.argmax(prof["J"]))
iP = int(np.argmax(prof["P"]))
iR = int(np.argmax(prof["rho_star"]))
print("\n[优化]")
print(f"   经济最优: S0={prof['S0'][iJ]:.0f} g/L, 放罐 {prof['t_harvest'][iJ]:.0f} h, "
      f"P={prof['P_harvest'][iJ]:.1f} g/L, J={prof['J'][iJ]:.5f} 元/(L·h)")
print(f"   产量最大: S0={prof['S0'][iP]:.0f}, P={prof['P'][iP]:.1f} g/L, t={prof['te'][iP]:.0f} h")
print(f"   生产强度最大: S0={prof['S0'][iR]:.0f}, rho={prof['rho_star'][iR]:.3f} g/(L·h) "
      f"@ t={prof['t_star'][iR]:.0f} h")

cost_prof = np.array([(COST["p_S"] * s + COST["c_op"] * prof["te"][i]
                       + COST["c_var"] * prof["P"][i] + COST["c_vol"])
                      / prof["P"][i] * 1000 for i, s in enumerate(grid_S0)])
iC = int(np.argmin(cost_prof))
print(f"   单位产品成本最低: S0={grid_S0[iC]:.0f} g/L, {cost_prof[iC]:.2f} 元/kg, "
      f"t={prof['te'][iC]:.0f} h, P={prof['P'][iC]:.1f} g/L")

sens_S = []
for c_vol in [0.04, 0.08, 0.12, 0.20, 0.32]:
    row = []
    for pP in [0.008, 0.010, 0.012, 0.015]:
        cc = dict(COST, c_vol=c_vol, p_P=pP)
        js = []
        for s in grid_S0:
            tt, Pv, te, Pe = trajectory(best.x, s)
            Jb = -1e18
            for th in np.arange(4.0, te + 1e-9, 1.0):
                J, _ = economics(s, th, cc, tt=tt, Pv=Pv)
                Jb = max(Jb, J)
            js.append(Jb)
        row.append(float(grid_S0[int(np.argmax(js))]))
    sens_S.append(row)
print("\n[敏感性] 最优初糖 (行 c_vol 元/L, 列 p_P 元/g):")
for cv, row in zip([0.04, 0.08, 0.12, 0.20, 0.32], sens_S):
    print(f"   c_vol={cv:.2f}: " + "  ".join(f"{v:5.0f}" for v in row))

C.json_dump(dict(
    table=tab,
    corr=dict(r_P=corr(S0v, PE), r_ravg=corr(S0v, RAVG), r_te=corr(S0v, TE),
              r_rmax=corr(S0v, RMAX)),
    end_models=ends,
    crit=dict(Y=Y_end, Pcap=Pcap_end, S0crit=float(S0crit)),
    od=dict(a=float(c_od[0]), b=float(c_od[1]), stats=st_od),
    rmax_model=dict(params=dict(zip(["Vmax", "Ks", "Ki"],
                                    [float(v) for v in r_rm.x])), stats=st_rm,
                    fit=rmax_haldane(r_rm.x, S0v).tolist()),
    kinetic=dict(params=dict(zip(nm6, [float(v) for v in best.x])),
                 se=dict(zip(nm6, [float(v) for v in se])),
                 r2=dict(te=r2_t["r2"], P=r2_P["r2"], rmax=r2_r["r2"], ravg=r2_ra["r2"]),
                 rmse=dict(te=r2_t["rmse"], P=r2_P["rmse"], rmax=r2_r["rmse"],
                           ravg=r2_ra["rmse"]),
                 pred=pred_tab, cost=float(best.cost)),
    cost_model=COST,
    profile={k: v.tolist() for k, v in prof.items()},
    unit_cost=cost_prof.tolist(),
    opt=dict(J=dict(S0=float(prof["S0"][iJ]), t=float(prof["t_harvest"][iJ]),
                    P=float(prof["P_harvest"][iJ]), J=float(prof["J"][iJ])),
             P=dict(S0=float(prof["S0"][iP]), P=float(prof["P"][iP]),
                    t=float(prof["te"][iP])),
             rho=dict(S0=float(prof["S0"][iR]), rho=float(prof["rho_star"][iR]),
                      t=float(prof["t_star"][iR])),
             cost=dict(S0=float(grid_S0[iC]), cost=float(cost_prof[iC]),
                       t=float(prof["te"][iC]), P=float(prof["P"][iC]))),
    sens=dict(c_vol=[0.04, 0.08, 0.12, 0.20, 0.32], p_P=[0.008, 0.010, 0.012, 0.015],
              S0=sens_S, grid_S0=grid_S0.tolist()),
), "q3_results.json")
print("\n已保存 results/q3_results.json")
