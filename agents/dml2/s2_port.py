"""Step 1b/2/4: reproduce the agents/ml sleeve gross, re-cost it under MOC + $3/order, sweep order-count reductions.
usage: python s2_port.py <grid> [pred tags comma list]"""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dml2")
from dlib import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)

def score(P, m, tags, sm=5):
    S = np.mean([xs_rank_rows(p, m & np.isfinite(p)) for p in (load_pred(t) for t in tags)], axis=0)
    S = np.where(m, S, np.nan)
    if sm > 1:
        S = pd.DataFrame(S).rolling(sm, min_periods=1).mean().to_numpy(); S = np.where(m & np.isfinite(S), S, np.nan)
    return S

def orig_style(P, F, m, S, beta, k=25, hold=5):
    """agents/ml/ml_port.build_W(k25, hold5, eq, buf0, hedge) + ls.backtest_W(nextopen) gross (decided close t, entered next open)."""
    T, N = P.T, P.N; spy = P.col("SPY"); tgt = np.zeros((T, N))
    for t in range(T):
        s = S[t]; idx = np.where(np.isfinite(s))[0]
        if len(idx) < 4 * k: continue
        o = idx[np.argsort(-s[idx])]; tgt[t, o[:k]] = 1 / k; tgt[t, o[-k:]] = -1 / k
    c = np.cumsum(tgt, axis=0); W = np.zeros((T, N)); W[hold:] = (c[hold:] - c[:-hold]) / hold
    W[:, spy] = -np.clip(np.nansum(W * beta, axis=1), -0.5, 0.5)
    cc = F["dret"]; oc = F["oc"]
    Wp = np.vstack([np.zeros((1, N)), W[:-1]]); Wpp = np.vstack([np.zeros((1, N)), Wp[:-1]])
    pr = np.nansum(Wpp * cc, axis=1) + np.nansum((Wp - Wpp) * oc, axis=1); turn = np.abs(Wp - Wpp).sum(axis=1)
    i0 = int(np.searchsorted(P.dates, pd.Timestamp(START)))
    g = pd.Series(pr[i0:], P.dates[i0:]); tu = pd.Series(turn[i0:], P.dates[i0:])
    return g, tu

if __name__ == "__main__":
    grid = sys.argv[1]
    P, F, m = setup(); aud = liq_mats(P); sim = Sim(P, F, m, aud)
    res = []
    log(f"liq coverage: share of universe (mask) cells with closing-auction data {np.isfinite(aud[m]).mean():.3f}")
    def go(name, S, **kw):
        t0 = time.time(); df = sim.run(S, **kw); r = summ(df, name); r["s"] = round(time.time() - t0); res.append(r)
        log(" ".join(f"{k}={v}" for k, v in r.items())); return df
    if grid == "repro":
        S = score(P, m, ["b1_r126", "b5_r126", "b10_r126"], 5)
        g, tu = orig_style(P, F, m, S, sim.beta.copy() * 0 + beta_mat(F))
        ref = pd.read_csv(f"{HERE}/agents/ml/out/gross_final2_k25h5b0H_sm5.csv", index_col=0, parse_dates=True).iloc[:, 0]
        refn = pd.read_csv(f"{HERE}/agents/ml/logs/net_lgb1n+lgb5n+lgb10n_k25_h5_d_eq_buf0_H_sm5.csv", index_col=0, parse_dates=True).iloc[:, 0]
        j = pd.concat([g, ref], axis=1, join="inner").loc[START:DEV_END]
        log(f"REPRO orig-style (next-open) gross Sh {rlib.sh(g):.2f} vs agents/ml gross {rlib.sh(ref.loc[START:DEV_END]):.2f}; corr {j.corr().iloc[0,1]:.3f}; turn/day {tu.mean():.3f}; "
            f"net@10bps {rlib.sh(g - tu*10e-4):.2f} (agents/ml net {rlib.sh(refn.loc[START:DEV_END]):.2f}); fold-OOS ref net {rlib.sh(rlib.fold_oos(refn, p=False)[0]):.2f}")
        pd.DataFrame({"mine": g, "ref": ref}).loc[START:DEV_END].to_csv(f"{OUT}/repro_gross.csv")
        for tm in ["B0", "A"]:
            df = go(f"orig_ens_tranche_k25h5_sm5_H_{tm}", S, k=25, mode="tranche", hold=5, hedge=True, timing=tm)
            if tm == "A": df.to_csv(f"{OUT}/sim_orig_A.csv")
        for c in [0.1, 1.0]: go(f"orig_ens_tranche_k25h5_sm5_H_A_c{c}", S, k=25, mode="tranche", hold=5, hedge=True, timing="A", c=c)
        go("orig_ens_tranche_k25h5_sm5_H_A_nocom", S, k=25, mode="tranche", hold=5, hedge=True, timing="A", order_usd=0)
    if grid in ("sweep", "sweep_s"):
        tags = sys.argv[2].split(","); sm = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        S = score(P, m, tags, sm); tagn = "+".join(tags) + f"_sm{sm}"
        for k in [15, 20, 25]:
            for hold in [5, 10]:
                go(f"{tagn} tranche k{k} h{hold}", S, k=k, mode="tranche", hold=hold)
            for reb in [5, 10]:
                go(f"{tagn} full k{k} reb{reb}", S, k=k, mode="band", reb=reb, buf=0)
            for reb in [1, 5, 10]:
                for buf in ([15, 30, 60] if reb == 1 else [15, 30]):
                    go(f"{tagn} band k{k} reb{reb} buf{buf}", S, k=k, mode="band", reb=reb, buf=buf)
        for mtu in [2000, 5000]:
            go(f"{tagn} tranche k25 h5 mt{mtu}", S, k=25, mode="tranche", hold=5, mt=mtu / CAP)
    pd.DataFrame(res).to_csv(f"{OUT}/res_{grid}_{len(sys.argv) > 2 and sys.argv[2].replace(',', '+')}.csv", index=False)
    print(pd.DataFrame(res).to_string())
