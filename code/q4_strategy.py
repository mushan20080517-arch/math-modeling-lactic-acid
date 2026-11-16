# -*- coding: utf-8 -*-
"""问题 4：过程强化策略的定量评估
选择策略：及时移出乳酸（原位产物分离 ISPR），并与补料分批（fed-batch）对比。
"""
from __future__ import annotations

import numpy as np

import common as C
from growth import f_logistic, f_gompertz, f_richards, f_malthus, f_malthus_lag

FUNCS = dict(f_logistic=f_logistic, f_gompertz=f_gompertz,
             f_richards=f_richards, f_malthus=f_malthus, f_malthus_lag=f_malthus_lag)

q1 = C.load_json("q1_results.json")
q2 = C.load_json("q2_results.json")
q3 = C.load_json("q3_results.json")

LP = np.array(q1[q1["best"]]["p"])
LF = FUNCS[q1[q1["best"]]["fun"]]
L_LOW = LP[0]
alpha = q2["best"]["params"]["alpha"]
beta = q2["best"]["params"]["beta"]
Pm = q2["best"]["params"]["Pm"]
Yps = q2["best"]["params"]["Yps"]
ms = q2["best"]["params"]["ms"]
S0 = C.S0_T1
t1 = C.T1_T

print("=" * 78)
print("问题4：过程强化策略的定量评估")
print("=" * 78)
print(f"模型参数: alpha={alpha:.4f} beta={beta:.5f} Pm={Pm:.4f} Yps={Yps:.4f} ms={ms:.4f}")


def Lof(x):
    return float(LF(LP, np.array([float(x)]))[0])


def dLof(x):
    A, k, tm = LP[1], LP[2], LP[3]
    e = np.exp(-k * (float(x) - tm))
    return k * A * e / (1.0 + e) ** 2


def Xof(x):
    return 10.0 ** (Lof(x) - L_LOW)


def prod_rate(t, P):
    return max(alpha * dLof(t) + beta * Xof(t), 0.0) * max(1.0 - P / Pm, 0.0)


def batch(tmax=30.0, dt=0.01):
    tt, Pv, Sv = [0.0], [0.0], [S0]
    t, P, S = 0.0, 0.0, S0
    while t < tmax:
        r = prod_rate(t, P) if S > 1e-9 else 0.0
        P += r * dt
        S = max(S - (r / Yps + ms) * dt, 0.0)
        t += dt
        tt.append(t), Pv.append(P), Sv.append(S)
    return np.array(tt), np.array(Pv), np.array(Sv)


# --------------------------------------------------------------------------
# 1. 基线：不控酸的分批发酵
# --------------------------------------------------------------------------
tb, Pb, Sb = batch(28.0)
i14 = int(np.argmin(np.abs(tb - 14.0)))
print(f"\n[基线] 14 h: P={Pb[i14]:.2f} g/L, S={Sb[i14]:.2f} g/L；"
      f"28 h: P={Pb[-1]:.2f} g/L, S={Sb[-1]:.2f} g/L")
batch_prod = float(Pb[i14])
batch_time = 14.0

# --------------------------------------------------------------------------
# 2. 原位移出乳酸：把罐内乳酸维持在设定水平 Psp
# --------------------------------------------------------------------------
def ispr(Psp, tmax=60.0, dt=0.005, s_end=0.5):
    t, Pv, S = 0.0, 0.0, S0
    produced = 0.0
    removed = 0.0
    Pvessel = 0.0
    rec = []
    while t < tmax and S > s_end:
        P_eff = min(Pvessel, Psp) if Pvessel > 0 else 0.0
        r = prod_rate(t, P_eff)
        if r < 0.02 and produced > 1e-6:
            break
        produced += r * dt
        Pvessel += r * dt
        if Pvessel > Psp:                    # 超出的部分被移出
            removed += (Pvessel - Psp)
            Pvessel = Psp
        S = max(S - (r / Yps + ms) * dt, 0.0)
        t += dt
        rec.append((t, Pvessel, produced, S))
    rec = np.array(rec)
    return dict(Psp=Psp, total=float(produced), removed=float(removed),
                t=float(t), S_end=float(S), S_used=float(S0 - S),
                rho=float(produced / t) if t > 0 else 0.0, traj=rec)

print("\n[策略: 原位移出乳酸] 不同控制水平下的效果")
print(" Psp  P总(g/L) 移除(g/L)  时长(h)  残余糖(g/L) 产出强度(g/(L·h)) 相对基线")
rows = []
base = float(Pb[i14])
for Psp in [2.0, 4.0, 6.0, 8.0, 9.0]:
    r = ispr(Psp)
    rows.append(dict(Psp=Psp, total=r["total"], removed=r["removed"], t=r["t"],
                     S_end=r["S_end"], rho=r["rho"], gain=r["total"] / base,
                     traj=r["traj"]))
    print(f" {Psp:4.1f}  {r['total']:8.2f} {r['removed']:8.2f} {r['t']:8.1f} "
          f"{r['S_end']:10.2f} {r['rho']:14.3f} {r['total']/base:9.2f}x")

print(f"\n基线: 14 h 产酸 {base:.2f} g/L, 产出强度 {base/14:.3f} g/(L·h), 残糖 {Sb[i14]:.2f} g/L")

# 不控制时"理论极限"（糖全部转化）
P_lim = Yps * S0
print(f"糖全部转化的理论上限: {P_lim:.2f} g/L")

# --------------------------------------------------------------------------
# 3. 对比策略：补料分批（用试验二的动力学模型）
# --------------------------------------------------------------------------
k3 = q3["kinetic"]["params"]
Vmax, Ks, Ki, Pcap, Y, vstop = (k3["Vmax"], k3["Ks"], k3["Ki"], k3["Pcap"],
                                k3["Y"], k3["vstop"])


def fed_batch(S0_0=40.0, S_set=30.0, S_feed=500.0, V0=1.0, total_sugar=115.0,
              dt=0.01, tmax=300.0, Pcap_eff=None):
    """保持罐内残糖为 S_set 的指数补料（简化：以反馈补料维持设定浓度）。"""
    Pc = Pcap if Pcap_eff is None else Pcap_eff
    V, S, P = V0, S0_0, 0.0
    fed = 0.0
    t = 0.0
    rec = []
    M = S0_0 * V0                     # 初始糖总量
    while t < tmax:
        rr = Vmax * S / (Ks + S + S * S / Ki) * max(1 - P / Pc, 0.0)
        if rr < vstop:
            break
        # 维持 S = S_set 所需补料（糖质量守恒）
        need = 0.0
        if S < S_set and fed + S0_0 * V0 < total_sugar:
            need = (S_set - S) * V          # 需要补入的糖质量
            need = min(need + rr * V * dt / Y, total_sugar - (fed + S0_0 * V0))
            need = max(need, 0.0)
        dV = need / S_feed
        Vn = V + dV
        S = (S * V + need) / Vn
        P = P * V / Vn                  # 体积稀释
        V = Vn
        fed += need
        P += rr * dt
        S = max(S - rr * dt / Y, 0.0)
        t += dt
        rec.append((t, V, S, P, fed))
    return dict(t=t, V=V, S=S, P=P, fed=fed, total_sugar=S0_0 * V0 + fed,
                mass=P * V, rho=P * V / (t * V0), traj=np.array(rec))


print("\n[对比策略: 补料分批]")
fb = fed_batch()
print(f"   初始 40 g/L + 补糖 {fb['fed']:.1f} g，总糖 {fb['total_sugar']:.1f} g，"
      f"终点 V={fb['V']:.3f} L, P={fb['P']:.2f} g/L, S={fb['S']:.3f} g/L")
print(f"   总产物 {fb['mass']:.1f} g，时长 {fb['t']:.1f} h，容积产率 {fb['rho']:.3f} g/(L·h)")
print(f"   对比：150 g/L 分批 → P={q3['opt']['P']['P']:.1f} g/L, t={q3['opt']['P']['t']:.0f} h；"
      f" 残糖 35.7 g/L，浪费糖 {150*Y - q3['opt']['P']['P']:.1f} g/L")

# 补料分批在 150 g/L 总糖下的理论收益：不浪费糖
print(f"\n   若按 150 g/L 总糖补料（不浪费）：理论上限 P={Y*150:.1f} g/L，"
      f"但受上限 Pcap={Pcap:.1f} 限制，实际仍是 {Pcap:.1f} g/L")

C.json_dump(dict(
    params=dict(alpha=alpha, beta=beta, Pm=Pm, Yps=Yps, ms=ms, L0=L_LOW),
    batch=dict(t=tb.tolist(), P=Pb.tolist(), S=Sb.tolist(),
               P14=float(Pb[i14]), S14=float(Sb[i14]),
               P28=float(Pb[-1]), S28=float(Sb[-1])),
    ispr=[dict(Psp=r["Psp"], total=r["total"], removed=r["removed"], t=r["t"],
               S_end=r["S_end"], rho=r["rho"], gain=r["gain"],
               traj=r["traj"].tolist()) for r in rows],
    fed_batch=dict(t=fb["t"], V=fb["V"], S=fb["S"], P=fb["P"], fed=fb["fed"],
                   total=fb["total_sugar"], mass=fb["mass"], rho=fb["rho"],
                   traj=fb["traj"].tolist()),
    P_limit=float(P_lim),
), "q4_results.json")
print("\n已保存 results/q4_results.json")
