#!/usr/bin/env python3
import numpy as np, pandas as pd, sys, time, itertools
from bt import *
from feats import *
from scipy.stats import rankdata

P = Panel(); F = build(P)
print("features built", flush=True)

def ranks(s, m):
    """rank within mask m, 0..1 (1 best); -inf outside"""
    out = np.full_like(s, -np.inf)
    idx = np.where(m & np.isfinite(s))[0]
    if len(idx):
        out[idx] = rankdata(s[idx]) / len(idx)
    return out

def make_fn(top=20, min_adv=50e6, score="vs", weight="eq", freq="m", hyst=2.0,
            voltarget=None, stop_ma=None, min_px=5.0, lb=252, skip=21):
    state = {"hold": np.array([], dtype=int)}
    def fn(i):
        m = base_mask(F, i, min_adv, min_px)
        v = F["vol60"][i] * np.sqrt(252)
        m12 = mom(F, i, lb, skip)
        if score == "vs":
            s = m12 / (v + 1e-9)
        elif score == "raw":
            s = m12
        elif score == "comp":   # composite rank
            s6 = mom(F, i, 126, skip) / (v + 1e-9)
            hi = F["closef"][i] / F["hi252"][i]
            s = ranks(m12 / (v + 1e-9), m) + ranks(s6, m) + ranks(hi, m)
        elif score == "comp2":
            s6 = mom(F, i, 126, skip) / (v + 1e-9)
            s = ranks(m12 / (v + 1e-9), m) + ranks(s6, m)
        s = np.where(m & np.isfinite(s), s, -np.inf)
        order = np.argsort(-s)
        order = order[np.isfinite(s[order])]
        if freq == "m" or hyst is None:
            idx = order[:top]
        else:
            keep_set = set(order[: int(top * hyst)].tolist())
            keep = [c for c in state["hold"] if c in keep_set]
            if stop_ma is not None:
                keep = [c for c in keep if F["closef"][i, c] > F[f"ma{stop_ma}"][i, c]]
            new = [c for c in order if c not in set(keep)][: max(0, top - len(keep))]
            idx = np.array(keep + new, dtype=int)
        state["hold"] = idx
        if len(idx) == 0:
            return {}
        if weight == "eq":
            w = np.full(len(idx), 1.0 / len(idx))
        else:
            iv = 1.0 / (v[idx] + 0.05); w = iv / iv.sum(); w = np.minimum(w, 0.1); w = w / w.sum()
        expo = 1.0
        if voltarget is not None:
            # realized vol of the equal-weight basket over last 20 days
            br = F["dret"][i - 20:i][:, idx].mean(axis=1)
            bv = br.std() * np.sqrt(252)
            expo = min(1.0, voltarget / (bv + 1e-9))
        return {int(c): float(x * expo) for c, x in zip(idx, w)}
    return fn

grid = []
for score, top, freq, hyst, vt, stop in itertools.product(
        ["vs", "comp", "comp2", "raw"], [20, 30], ["m", "w"], [1.5, 2.5], [None, 0.25], [None, 100]):
    if freq == "m" and (hyst != 1.5):
        continue
    if freq == "m" and stop is not None:
        continue
    grid.append(dict(score=score, top=top, freq=freq, hyst=hyst, voltarget=vt, stop_ma=stop))
print(len(grid), "configs", flush=True)
res = []
for c in grid:
    fn = make_fn(**c)
    reb = F["rebal_m"] if c["freq"] == "m" else F["rebal_w"]
    eq, info = run_fast(P, fn, reb, START, END, cost_bps=15)
    s = stats(eq)
    # sub-period CAGRs
    e1 = eq[:"2021-10-27"]; e2 = eq["2021-10-27":]
    c1 = stats(e1)["cagr"]; c2 = stats(e2)["cagr"]
    res.append(dict(**c, cagr=s["cagr"], sharpe=s["sharpe"], maxdd=s["maxdd"], vol=s["vol"], turn=info["turnover"], cagr_h1=c1, cagr_h2=c2))
    print(f"{c}  CAGR {s['cagr']*100:.1f}% Sh {s['sharpe']:.2f} DD {s['maxdd']*100:.0f}% h1 {c1*100:.0f}% h2 {c2*100:.0f}% turn {info['turnover']:.0f}", flush=True)
df = pd.DataFrame(res).sort_values("cagr", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.to_string())
df.to_csv("logs/strat3_grid.csv", index=False)
