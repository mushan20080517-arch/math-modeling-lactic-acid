# -*- coding: utf-8 -*-
"""统一绘图脚本：由 results/*.json 生成论文插图。"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import common as C

plt = C.set_style()
q1 = C.load_json("q1_results.json")
q2 = C.load_json("q2_results.json")
q3 = C.load_json("q3_results.json")
q4 = C.load_json("q4_results.json")

t = C.T1_T
L, P, S, SC = C.T1_L, C.T1_P, C.T1_S, C.T1_SC
S0v, TE, PE, SR, RAVG, RMAX = C.T2_S0, C.T2_TE, C.T2_P, C.T2_SR, C.T2_RAVG, C.T2_RMAX


# ==========================================================================
# 图 1  数据总览
# ==========================================================================
def fig1():
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))

    a = ax[0, 0]
    a.plot(t, L, "o-", color=C.C_BLUE, lw=1.6, ms=5, label="菌体 lg(CFU/mL)")
    a.set_xlabel("发酵时间 / h")
    a.set_ylabel("菌体浓度 lg(CFU/mL)", color=C.C_BLUE)
    a.tick_params(axis="y", colors=C.C_BLUE)
    a2 = a.twinx()
    a2.plot(t, P, "s--", color=C.C_RED, lw=1.6, ms=5, label="乳酸 g/L")
    a2.set_ylabel("乳酸浓度 / (g/L)", color=C.C_RED)
    a2.tick_params(axis="y", colors=C.C_RED)
    a2.spines["right"].set_visible(True)
    a.set_title("(a) 试验一：菌体与乳酸")
    a.axvspan(0, 4, color="gray", alpha=0.10)
    a.axvspan(4, 11, color=C.C_GREEN, alpha=0.07)
    a.axvspan(11, 14, color="orange", alpha=0.08)
    a.set_ylim(7.5, 9.95)
    a.text(1.9, 9.78, "迟滞", ha="center", fontsize=9, color="#444")
    a.text(7.5, 9.78, "对数", ha="center", fontsize=9, color="#444")
    a.text(12.5, 9.78, "稳定", ha="center", fontsize=9, color="#444")

    b = ax[0, 1]
    b.plot(t, SC, "o-", color=C.C_GREEN, lw=1.8, ms=5, label="累计耗糖")
    b.plot(t, P, "s-", color=C.C_RED, lw=1.8, ms=5, label="乳酸")
    b.plot(t, S, "^-", color=C.C_BLUE, lw=1.8, ms=5, label="残糖")
    b.set_xlabel("发酵时间 / h")
    b.set_ylabel("浓度 / (g/L)")
    b.set_title("(b) 试验一：糖与酸的消长")
    b.legend(ncol=1, loc="center left", bbox_to_anchor=(0.02, 0.45))
    b.annotate("残糖仍有 24.9 g/L", xy=(14, 24.9), xytext=(7.0, 19.0),
               arrowprops=dict(arrowstyle="->", color="#666"), fontsize=9, color="#444")

    c = ax[1, 0]
    c.plot(S0v, PE, "o-", color=C.C_RED, lw=1.8, ms=6, label="终点乳酸")
    c.plot(S0v, S0v, ":", color="gray", lw=1.2, label="理论上限 (1 g/g)")
    c.set_xlabel("初始葡萄糖浓度 / (g/L)")
    c.set_ylabel("终点乳酸浓度 / (g/L)")
    c.set_title("(c) 试验二：终点产量")
    c.legend(loc="lower right")
    c.set_ylim(10, 135)
    c.annotate("残糖 35.7 g/L", xy=(150, 109.9), xytext=(92, 122),
               arrowprops=dict(arrowstyle="->", color="#666"), fontsize=9, color="#444")

    d = ax[1, 1]
    d.plot(S0v, RAVG, "o-", color=C.C_BLUE, lw=1.8, ms=6, label="平均产酸速率")
    d.plot(S0v, RMAX, "s--", color=C.C_GRAY, lw=1.5, ms=5, label="最大产酸速率")
    d.set_xlabel("初始葡萄糖浓度 / (g/L)")
    d.set_ylabel("产酸速率 / (g/(L·h))")
    d.set_title("(d) 试验二：速率随初糖的变化")
    d.legend(loc="upper right")
    d.set_ylim(0.6, 3.8)

    fig.tight_layout()
    return C.save(fig, "fig1_data.png")


# ==========================================================================
# 图 2  问题 1
# ==========================================================================
def fig2():
    fig = plt.figure(figsize=(11.4, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.26)

    a = fig.add_subplot(gs[0, :])
    tg = np.linspace(0, 16, 400)
    from growth import MODELS, f_logistic
    styles = {"M3 Logistic 模型": (C.C_RED, "-", "Logistic（选用）"),
              "M4 Gompertz 模型": (C.C_GREEN, "--", "Gompertz"),
              "M1 指数模型": (C.C_GRAY, ":", "指数模型")}
    for name, (col, ls, lab) in styles.items():
        spec = MODELS[name]
        pr = np.array(q1[name]["p"])
        a.plot(tg, spec["fun"](pr, tg), ls, color=col, lw=2.0, label=lab)
    a.plot(t, L, "o", color="k", ms=6.5, mfc="white", mew=1.6, label="实测值", zorder=5)
    a.set_xlabel("发酵时间 / h")
    a.set_ylabel("菌体浓度 lg(CFU/mL)")
    a.set_title("(a) 三种生长模型的拟合效果对比")
    a.legend(ncol=2, loc="lower right")
    a.set_xlim(-0.3, 15.8)
    p = np.array(q1["M3 Logistic 模型"]["p"])
    a.annotate(f"$L_{{max}}$={q1['M3 Logistic 模型']['Linf']:.2f}",
               xy=(14.6, p[0] + p[1]), xytext=(12.0, 8.55),
               arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9.5)

    b = fig.add_subplot(gs[1, 0])
    res = L - np.array(q1["M3 Logistic 模型"]["fit"])
    b.axhline(0, color="gray", lw=1)
    b.vlines(t, 0, res, color=C.C_BLUE, lw=1.4)
    b.plot(t, res, "o", color=C.C_BLUE, ms=5)
    b.set_xlabel("发酵时间 / h")
    b.set_ylabel("残差 / lg(CFU/mL)")
    b.set_title("(b) Logistic 模型残差")
    b.set_ylim(-0.06, 0.06)

    c = fig.add_subplot(gs[1, 1])
    spec = MODELS["M3 Logistic 模型"]
    v = spec["fun"](p, tg)
    d = np.gradient(v, tg)
    c.plot(tg, d, "-", color=C.C_RED, lw=2.0)
    fd = np.array(q1["_modelfree"]["fd_rate"])
    c.plot(t, np.clip(fd, 0, None), "o", color="k", ms=5, mfc="white", label="差分估计")
    i = int(np.argmax(d))
    c.plot(tg[i], d[i], "*", color=C.C_RED, ms=14, zorder=5)
    c.annotate(f"$\\mu_m$={d[i]:.3f}\n@ t={tg[i]:.1f} h", xy=(tg[i], d[i]),
               xytext=(8.6, d[i] * 0.72),
               arrowprops=dict(arrowstyle="->", color=C.C_RED), fontsize=9.5)
    c.set_xlabel("发酵时间 / h")
    c.set_ylabel("dL/dt / (lg·h$^{-1}$)")
    c.set_title("(c) 比生长速率曲线")
    c.legend(loc="upper left")

    return C.save(fig, "fig2_q1.png")


# ==========================================================================
# 图 3  问题 2
# ==========================================================================
def fig3():
    fig = plt.figure(figsize=(11.6, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
    tt = np.array([r["t"] for r in q2["table"]])
    Pf = np.array(q2["best"]["P"])
    Sf = np.array(q2["best"]["S"])

    a = fig.add_subplot(gs[0, 0])
    a.plot(SC, P, "o", color="k", ms=6, mfc="white", mew=1.6, label="实测")
    xs = np.linspace(0, 10.5, 50)
    a.plot(xs, q2["yield_model"]["Yps"] * xs, "-", color=C.C_RED, lw=2,
           label=f"P={q2['yield_model']['Yps']:.3f}·$S_c$")
    a.set_xlabel("累计葡萄糖消耗 $S_c$ / (g/L)")
    a.set_ylabel("乳酸浓度 / (g/L)")
    a.set_title("(a) 产酸—耗糖化学计量关系")
    a.legend(loc="upper left")
    a.text(7.4, 2.2, f"$R^2$={q2['yield_model']['stats']['r2']:.4f}", fontsize=9.5, color="#444")

    b = fig.add_subplot(gs[0, 1])
    tg = np.linspace(0, 14, 300)
    from growth import MODELS
    pL = np.array(q1[q1["best"]]["p"])
    b.plot(t, P, "o", color=C.C_RED, ms=6, mfc="white", mew=1.6, label="乳酸实测")
    b.plot(tt, Pf, "-", color=C.C_RED, lw=2)
    b.plot(t, S, "s", color=C.C_BLUE, ms=5.5, mfc="white", mew=1.5, label="残糖实测")
    b.plot(tt, Sf, "--", color=C.C_BLUE, lw=2)
    b.set_xlabel("发酵时间 / h")
    b.set_ylabel("浓度 / (g/L)")
    b.set_title("(b) 耦合动力学模型拟合")
    b.legend(loc="center right")
    b.text(0.5, 4.4, "实线/虚线：模型预测\n$R^2_P$=%.4f  $R^2_S$=%.4f"
           % (q2["best"]["statsP"]["r2"], q2["best"]["statsS"]["r2"]), fontsize=9, color="#444")

    c = fig.add_subplot(gs[1, 0])
    dPdt = np.array(q2["dPdt"])
    dLdt = np.array(q2["dLdt"])
    x = np.arange(len(t))
    c.bar(x - 0.2, dPdt, 0.4, color=C.C_RED, label="实测 dP/dt")
    tg_f = np.linspace(0, 14, 141)
    from scipy.integrate import solve_ivp  # noqa: F401
    Pfg = np.interp(tg_f, tt, Pf)
    mod = np.gradient(Pfg, tg_f)[::20]
    c.bar(x + 0.2, mod, 0.4, color=C.C_BLUE, alpha=0.75, label="模型 dP/dt")
    c.set_xticks(x)
    c.set_xticklabels([f"{int(v)}" for v in t])
    c.set_xlabel("发酵时间 / h")
    c.set_ylabel("dP/dt / (g·L$^{-1}$·h$^{-1}$)")
    c.set_title("(c) 产酸速率的分段对比")
    c.legend(loc="upper left")

    d = fig.add_subplot(gs[1, 1])
    d.plot(t, L, "o", color="k", ms=5, mfc="white", mew=1.4)
    X = np.array(q2["X_data"])
    ax2 = d.twinx()
    ax2.semilogy(t, X, "s-", color=C.C_GREEN, ms=4.5, lw=1.6)
    ax2.set_ylabel("相对菌体量 $X=10^{L-L_0}$", color=C.C_GREEN)
    ax2.tick_params(axis="y", colors=C.C_GREEN)
    ax2.spines["right"].set_visible(True)
    d.set_xlabel("发酵时间 / h")
    d.set_ylabel("lg(CFU/mL)")
    d.set_title("(d) 菌体量在 14 h 内增长约 112 倍")
    d.text(1.0, 9.25, "对数坐标", fontsize=9, color="#444")

    return C.save(fig, "fig3_q2.png")


# ==========================================================================
# 图 4  问题 3 终点规律
# ==========================================================================
def arr(d):
    return {k: np.array(v) for k, v in d.items()}


def fig4():
    prof = arr(q3["profile"])
    Sg = prof["S0"]
    k3 = q3["kinetic"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))

    a = ax[0, 0]
    a.plot(S0v, PE, "o", color=C.C_RED, ms=6.5, mfc="white", mew=1.7, label="实测（试验二）")
    a.plot(Sg, prof["P"], "-", color=C.C_RED, lw=2, label="产率—上限模型")
    Y, Pc, Sc = q3["crit"]["Y"], q3["crit"]["Pcap"], q3["crit"]["S0crit"]
    a.axvline(Sc, color=C.C_GRAY, ls=":", lw=1.4)
    a.text(Sc + 3, 26, f"$S_0^*$={Sc:.0f} g/L\n（糖/酸平衡点）", fontsize=9, color="#444")
    a.set_xlabel("初始葡萄糖浓度 / (g/L)")
    a.set_ylabel("终点乳酸 / (g/L)")
    a.set_title("(a) 终点产量：先线性后饱和")
    a.legend(loc="upper left")

    b = ax[0, 1]
    b.plot(S0v, TE, "o", color=C.C_BLUE, ms=6.5, mfc="white", mew=1.7, label="实测")
    b.plot(Sg, prof["te"], "-", color=C.C_BLUE, lw=2, label="动力学模型")
    b.set_xlabel("初始葡萄糖浓度 / (g/L)")
    b.set_ylabel("发酵终点时间 / h")
    b.set_title("(b) 时长随初糖超线性增长")
    b.legend(loc="upper left")
    b.text(24, 82, f"$R^2$={k3['r2']['te']:.3f}", fontsize=9.5, color="#444")

    c = ax[1, 0]
    c.plot(S0v, RAVG, "o", color=C.C_PURPLE, ms=6.5, mfc="white", mew=1.7, label="实测平均速率")
    c.plot(Sg, prof["P"] / prof["te"], "-", color=C.C_PURPLE, lw=2, label="模型平均速率")
    c.plot(Sg, prof["rho"], "--", color=C.C_ORANGE, lw=1.6,
           label="含辅助时间 $P/(t+t_0)$")
    c.set_xlabel("初始葡萄糖浓度 / (g/L)")
    c.set_ylabel("产酸速率 / (g/(L·h))")
    c.set_title("(c) 平均产酸速率单调下降")
    c.legend(loc="upper right")
    c.set_ylim(0.6, 2.6)

    d = ax[1, 1]
    d.plot(S0v, SR, "o", color=C.C_GREEN, ms=6.5, mfc="white", mew=1.7, label="实测残糖")
    d.plot(Sg, np.maximum(Sg - prof["P"] / Y, 0), "-", color=C.C_GREEN, lw=2,
           label="模型 $S_0-P/Y$")
    d.set_xlabel("初始葡萄糖浓度 / (g/L)")
    d.set_ylabel("终点残糖 / (g/L)")
    d.set_title("(d) 高糖条件下残糖陡增")
    d.legend(loc="upper left")
    d.axvspan(Sc, 155, color=C.C_RED, alpha=0.06)
    d.text(118, 12, "过量加糖区\n（糖被浪费）", fontsize=9, color="#a33")

    fig.tight_layout()
    return C.save(fig, "fig4_q3_endpoints.png")


# ==========================================================================
# 图 5  问题 3 优化
# ==========================================================================
def fig5():
    prof = arr(q3["profile"])
    Sg = prof["S0"]
    fig = plt.figure(figsize=(11.6, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30)

    a = fig.add_subplot(gs[0, 0])
    sc = a.scatter(prof["P"], prof["rho"], c=Sg, cmap="viridis", s=26)
    a.set_xlabel("终点产量（效价）/ (g/L)")
    a.set_ylabel("生产强度 $P/(t+t_0)$ / (g/(L·h))")
    a.set_title("(a) 产量—强度权衡（颜色=初糖）")
    cb = fig.colorbar(sc, ax=a, pad=0.02)
    cb.set_label("$S_0$ / (g/L)", fontsize=9)
    i = int(np.argmax(prof["J"]))
    a.plot(prof["P"][i], prof["rho"][i], "*", color=C.C_RED, ms=15,
           label=f"经济最优点 $S_0$={Sg[i]:.0f}")
    a.legend(loc="upper right")

    b = fig.add_subplot(gs[0, 1])
    unit_cost = np.array(q3["unit_cost"])
    b.plot(Sg, unit_cost, "-", color=C.C_RED, lw=2.2, label="单位产品成本")
    ax2 = b.twinx()
    ax2.plot(Sg, prof["J"], "--", color=C.C_BLUE, lw=1.8, label="单位时间收益 J")
    ax2.set_ylabel("J / (元·L$^{-1}$·h$^{-1}$)", color=C.C_BLUE)
    ax2.tick_params(axis="y", colors=C.C_BLUE)
    ax2.spines["right"].set_visible(True)
    b.set_xlabel("初始葡萄糖浓度 / (g/L)")
    b.set_ylabel("单位产品成本 / (元/kg)", color=C.C_RED)
    b.tick_params(axis="y", colors=C.C_RED)
    b.set_title("(b) 经济指标随初糖的变化")
    ic = int(np.argmin(unit_cost))
    b.plot(Sg[ic], unit_cost[ic], "o", color=C.C_RED, ms=8)
    b.annotate(f"{Sg[ic]:.0f} g/L\n{unit_cost[ic]:.2f} 元/kg",
               xy=(Sg[ic], unit_cost[ic]), xytext=(Sg[ic] + 8, unit_cost[ic] + 1.2),
               fontsize=9, color="#333", arrowprops=dict(arrowstyle="->", color="#666"))

    c = fig.add_subplot(gs[1, 0])
    Sm = np.array(q3["sens"]["S0"])
    im = c.imshow(Sm, cmap="YlGnBu", aspect="auto", origin="lower")
    c.set_xticks(range(len(q3["sens"]["p_P"])))
    c.set_xticklabels([f"{v:.3f}" for v in q3["sens"]["p_P"]])
    c.set_yticks(range(len(q3["sens"]["c_vol"])))
    c.set_yticklabels([f"{v:.2f}" for v in q3["sens"]["c_vol"]])
    c.set_xlabel("乳酸价格 $p_P$ / (元/g)")
    c.set_ylabel("下游体积成本 $c_{vol}$ / (元/L)")
    c.set_title("(c) 最优初糖对成本参数的敏感性")
    for i in range(Sm.shape[0]):
        for j in range(Sm.shape[1]):
            c.text(j, i, f"{Sm[i, j]:.0f}", ha="center", va="center", fontsize=9,
                   color="white" if Sm[i, j] > 85 else "#222")
    c.grid(False)
    fig.colorbar(im, ax=c, pad=0.02).set_label("最优 $S_0$ / (g/L)", fontsize=9)

    d = fig.add_subplot(gs[1, 1])
    kp = q3["kinetic"]["params"]
    cm = q3["cost_model"]

    def rate_s(S0x, Pp):
        Ss = S0x - Pp / kp["Y"]
        if Ss <= 0:
            return 0.0
        return (kp["Vmax"] * Ss / (kp["Ks"] + Ss + Ss * Ss / kp["Ki"])
                * max(1 - Pp / kp["Pcap"], 0.0))

    S0g = np.linspace(20, 150, 66)
    tg = np.arange(4.0, 150.0, 2.0)
    Jm = np.full((len(tg), len(S0g)), np.nan)
    for j, s0 in enumerate(S0g):
        dt = 0.1
        ts, Ps = [0.0], [0.0]
        p = 0.0
        while ts[-1] < 150.0 and rate_s(s0, p) > 0.12:
            p += rate_s(s0, p) * dt
            ts.append(ts[-1] + dt)
            Ps.append(p)
        Ps = np.array(Ps)
        for i, th in enumerate(tg):
            if th > ts[-1]:
                continue
            Pt = float(np.interp(th, ts, Ps))
            Jm[i, j] = (cm["p_P"] * Pt - cm["p_S"] * s0 - cm["c_var"] * Pt
                        - cm["c_vol"] - cm["c_op"] * th) / (th + cm["t_turn"])
    cs = d.contourf(S0g, tg, Jm, levels=16, cmap="RdYlGn")
    fig.colorbar(cs, ax=d, pad=0.02).set_label("J / (元·L$^{-1}$·h$^{-1}$)", fontsize=9)
    ij = np.unravel_index(np.nanargmax(Jm), Jm.shape)
    d.plot(S0g[ij[1]], tg[ij[0]], "*", color="k", ms=15)
    d.annotate(f"最优: $S_0$={S0g[ij[1]]:.0f} g/L,\n放罐 {tg[ij[0]]:.0f} h",
               xy=(S0g[ij[1]], tg[ij[0]]), xytext=(S0g[ij[1]] + 14, tg[ij[0]] + 30),
               fontsize=9, color="k",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
               arrowprops=dict(arrowstyle="->", color="k"))
    d.set_ylim(140, 0)
    d.set_xlabel("初始葡萄糖浓度 $S_0$ / (g/L)")
    d.set_ylabel("放罐时间 / h")
    d.set_title("(d) 初糖—放罐时间的联合优化")

    return C.save(fig, "fig5_q3_opt.png")


# ==========================================================================
# 图 6  问题 4
# ==========================================================================
def fig6():
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.9))
    bt = np.array(q4["batch"]["t"])
    bP = np.array(q4["batch"]["P"])
    bS = np.array(q4["batch"]["S"])

    a = ax[0]
    a.plot(bt[bt <= 20], bP[bt <= 20], "-", color=C.C_GRAY, lw=2.2, label="分批（不控酸）")
    mk = {2.0: "-", 4.0: "--", 6.0: "-."}
    for r in q4["ispr"]:
        if r["Psp"] in (2.0, 4.0, 8.0):
            tr = np.array(r["traj"])
            a.plot(tr[:, 0], tr[:, 2], mk.get(r["Psp"], ":"), lw=1.9,
                   label=f"移酸，维持 {r['Psp']:.0f} g/L")
    a.set_xlim(0, 20)
    a.set_xlabel("发酵时间 / h")
    a.set_ylabel("累计产酸 / (g/L)")
    a.set_title("(a) 移出乳酸大幅提高单批产量")
    a.legend(loc="lower right")

    b = ax[1]
    Psp = [r["Psp"] for r in q4["ispr"]]
    rho = [r["rho"] for r in q4["ispr"]]
    b.plot(Psp, rho, "o-", color=C.C_RED, lw=2, ms=7)
    b.axhline(q4["batch"]["P14"] / 14, color=C.C_GRAY, ls="--", lw=1.4)
    b.text(2.2, q4["batch"]["P14"] / 14 + 0.08, "分批基线", fontsize=9, color="#555")
    b.set_xlabel("罐内乳酸控制水平 / (g/L)")
    b.set_ylabel("平均产出强度 / (g/(L·h))")
    b.set_title("(b) 控制水平 vs 产出强度")

    c = ax[2]
    tr = np.array(q4["fed_batch"]["traj"])
    c.plot(tr[:, 0], tr[:, 3], "-", color=C.C_BLUE, lw=2, label="补料分批 P")
    c.plot(tr[:, 0], tr[:, 2], "-", color=C.C_GREEN, lw=1.6, label="罐内残糖 S")
    ax2 = c.twinx()
    ax2.plot(tr[:, 0], tr[:, 1], ":", color=C.C_GRAY, lw=1.4)
    ax2.set_ylabel("装液体积 / L", color=C.C_GRAY)
    ax2.tick_params(axis="y", colors=C.C_GRAY)
    ax2.spines["right"].set_visible(True)
    c.set_xlabel("发酵时间 / h")
    c.set_ylabel("浓度 / (g/L)")
    c.set_title("(c) 补料分批：不再浪费糖")
    c.legend(loc="center right")

    fig.tight_layout()
    return C.save(fig, "fig6_q4.png")


# ==========================================================================
# 图 7  建模流程
# ==========================================================================
def fig7():
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    boxes = [
        (0.15, 2.9, 1.55, 1.05, "读题与拆解\n明确变量/目标", "#ebf4ff", C.C_BLUE),
        (1.95, 2.9, 1.55, 1.05, "数据整理\n画图找形态", "#ebf4ff", C.C_BLUE),
        (3.75, 2.9, 1.55, 1.05, "选模型\n机理+经验", "#fff5eb", C.C_ORANGE),
        (5.55, 2.9, 1.55, 1.05, "参数估计\n最小二乘", "#fff5eb", C.C_ORANGE),
        (7.35, 2.9, 1.75, 1.05, "检验\nR²/残差/外推", "#f0fff4", C.C_GREEN),
        (5.55, 1.35, 1.55, 1.05, "灵敏度\n与稳健性", "#f0fff4", C.C_GREEN),
        (3.75, 1.35, 1.55, 1.05, "优化与决策\n权衡取舍", "#faf5ff", C.C_PURPLE),
        (1.95, 1.35, 1.55, 1.05, "写作\n图表+结论", "#faf5ff", C.C_PURPLE),
    ]
    for (x, y, w, h, txt, fc, ec) in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.6, color="#222")
    arrows = [((1.72, 3.42), (1.93, 3.42)), ((3.52, 3.42), (3.73, 3.42)),
              ((5.32, 3.42), (5.53, 3.42)), ((7.12, 3.42), (7.33, 3.42)),
              ((8.22, 2.88), (7.12, 2.42)), ((6.30, 2.42), (6.30, 2.40)),
              ((5.53, 1.87), (5.32, 1.87)), ((3.73, 1.87), (3.52, 1.87))]
    for (p1, p2) in arrows:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                     color="#555", lw=1.3))
    ax.text(8.25, 2.35, "不改模型？\n回头调整", fontsize=8.8, color="#666", ha="center")
    ax.text(5.0, 0.75, "本题主线：问题1 生长 → 问题2 生长-产酸-耗糖耦合 → 问题3 初糖优化 → 问题4 过程强化",
            ha="center", fontsize=10, color="#333")
    ax.add_patch(FancyBboxPatch((0.9, 0.45), 8.2, 0.62, boxstyle="round,pad=0.04",
                                fc="#f7fafc", ec="#cbd5e0", lw=1.0))
    ax.text(5.0, 0.76, "本题主线：问题1 生长 → 问题2 生长-产酸-耗糖耦合 → 问题3 初糖优化 → 问题4 过程强化",
            ha="center", va="center", fontsize=10, color="#333")
    return C.save(fig, "fig7_workflow.png")


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        print("saved", f())
