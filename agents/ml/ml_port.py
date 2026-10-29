#!/usr/bin/env python3
"""Portfolio construction on cached predictions. Usage:
  python agents/ml/ml_port.py --pred lgb5[,lgb10,...] [--min_adv 50e6] [--cost 10]
Evaluates grid of (topK, holding/tranches, rebalance, hysteresis, beta hedge)."""
import numpy as np, pandas as pd, sys, os, time, argparse
sys.path.insert(0, "/home/trumpet/strat20"); sys.path.insert(0, "/home/trumpet/strat20/agents/ml")
from mlfeat import *

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True); ap.add_argument("--min_adv", type=float, default=50e6)
ap.add_argument("--cost", type=float, default=10.0); ap.add_argument("--borrow", type=float, default=100.0)
ap.add_argument("--grid", default="main"); ap.add_argument("--save", default="")
ap.add_argument("--topn_univ", type=int, default=0, help="restrict to N most liquid names each day")
ap.add_argument("--smooth", type=int, default=1, help="average the daily rank score over the last n days")
ap.add_argument("--hedge_cap", type=float, default=0.5, help="cap on |SPY hedge| as fraction of NAV")
ap.add_argument("--short_adv", type=float, default=0, help="shorts only in names with adv20 >= this")
ap.add_argument("--short_px", type=float, default=0, help="shorts only in names with raw close >= this")
ap.add_argument("--wsave", default="", help="save weights of the best config to out/W_<wsave>.npy")
args = ap.parse_args()
OUT = "/home/trumpet/strat20/agents/ml/out"

P = load_sub(args.min_adv)
F = build(P); F = extra_feats(P, F)
m = build_mask(F, P, args.min_adv)
if args.topn_univ:
    adv = np.where(m, F["adv20"], -1)
    thr = -np.sort(-adv, axis=1)[:, min(args.topn_univ, P.N) - 1][:, None]
    m = m & (adv >= thr)
T, N = P.T, P.N
preds = []
for tg in args.pred.split(","):
    p = np.load(f"{OUT}/pred_{tg}.npy")
    preds.append(xs_rank_rows(p, m & np.isfinite(p)))
S = np.mean(preds, axis=0); S = np.where(m & np.isfinite(S), S, np.nan)
if args.smooth > 1:
    S = pd.DataFrame(S).rolling(args.smooth, min_periods=1).mean().values
    S = np.where(m & np.isfinite(S), S, np.nan)
# beta for hedging
dret = F["dret"]; spy = F["spy_ret"]; W = 252
rm = lambda a, w: pd.DataFrame(a).rolling(w, min_periods=200).mean().values
mx = pd.Series(spy).rolling(W, min_periods=200).mean().values[:, None]
beta = np.clip(np.nan_to_num((rm(dret * spy[:, None], W) - mx * rm(dret, W)) / (pd.Series(spy).rolling(W, min_periods=200).var().values[:, None] + 1e-12)), -1, 4)
spy_col = P.col("SPY")
i0 = int(np.searchsorted(P.dates, pd.Timestamp(START))); i1 = int(np.searchsorted(P.dates, pd.Timestamp(END), side="right"))

rawc = ffill(P.close_raw, 5)
VOL = np.nan_to_num(F["vol60"], nan=0.03); _T = 0
def target_weights(s_row, k, mode="eq", buf=0, prev=None, sok=None):
    if sok is not None:   # restricted short universe: longs from the full set, shorts among eligible names only
        wl = target_weights(s_row, k, mode, buf, prev); wl[wl < 0] = 0
        ws = target_weights(np.where(sok, s_row, np.nan), k, mode, buf, prev); ws[ws > 0] = 0
        return wl + ws
    idx = np.where(np.isfinite(s_row))[0]
    w = np.zeros(N)
    if len(idx) < 4 * k: return w
    o = idx[np.argsort(-s_row[idx])]
    if buf and prev is not None:
        # hysteresis: keep existing longs if still within top k+buf, shorts within bottom k+buf
        rank = np.full(N, 10 ** 9); rank[o] = np.arange(len(o))   # names outside the universe are dropped
        keepL = [c for c in np.where(prev > 0)[0] if rank[c] < k + buf]
        keepS = [c for c in np.where(prev < 0)[0] if rank[c] < 10 ** 9 and rank[c] >= len(o) - k - buf]
        L = list(keepL) + [c for c in o if c not in set(keepL) and c not in set(keepS)][: max(0, k - len(keepL))]
        Sx = list(keepS) + [c for c in o[::-1] if c not in set(keepS) and c not in set(L)][: max(0, k - len(keepS))]
    else:
        L = o[:k]; Sx = o[-k:][::-1]
    if mode == "eq":
        w[L] = 1.0 / len(L); w[Sx] = -1.0 / len(Sx)
    elif mode == "iv":   # inverse-vol within each leg
        vl = 1.0 / np.clip(VOL[_T, L], 0.005, 0.2); w[L] = vl / vl.sum()
        vs = 1.0 / np.clip(VOL[_T, Sx], 0.005, 0.2); w[Sx] = -vs / vs.sum()
    else:  # rank-weighted within top/bottom
        rl = np.arange(len(L), 0, -1, dtype=float); w[L] = rl / rl.sum()
        Sx = sorted(Sx, key=lambda c: s_row[c])   # worst first
        rs = np.arange(len(Sx), 0, -1, dtype=float); w[Sx] = -rs / rs.sum()
    return w

def build_W(k, hold, reb="d", mode="eq", buf=0, hedge=False):
    global _T
    """hold = number of overlapping tranches (days each tranche is held). reb 'd' daily or 'w' weekly."""
    Wt = np.zeros((T, N)); prev = None
    rset = set(F["rebal_w"]) if reb == "w" else None
    tgt = np.zeros((T, N)); last = np.zeros(N)
    sokm = None
    if args.short_adv or args.short_px:
        sokm = (F["adv20"] >= args.short_adv) & (rawc >= args.short_px)
    for t in range(i0 - hold - 10, i1):
        if reb == "w" and t not in rset:
            tgt[t] = last; continue
        _T = t
        last = target_weights(S[t], k, mode, buf, prev, None if sokm is None else sokm[t]); prev = last if buf else None
        tgt[t] = last
    if hold > 1 and reb == "d":
        c = np.cumsum(tgt, axis=0); Wt[hold:] = (c[hold:] - c[:-hold]) / hold; Wt[:hold] = c[:hold] / hold
    else:
        Wt = tgt
    if hedge:
        b = np.nansum(Wt * beta, axis=1); Wt[:, spy_col] = -np.clip(b, -args.hedge_cap, args.hedge_cap)
    return Wt

def evaluate(name, Wt, exec="nextopen"):
    net, gross, info = backtest_W(P, F, Wt, cost_bps=args.cost, borrow_bps=args.borrow, exec=exec)
    yr = net.groupby(net.index.year).apply(sh)
    eq = (1 + net).cumprod(); dd = (eq / eq.cummax() - 1).min()
    print(f"{name:40s} net {sh(net):5.2f} gross {sh(gross):5.2f} ret {net.mean()*252*100:6.1f}% vol {net.std()*np.sqrt(252)*100:5.1f}% DD {dd*100:6.1f}% turn {info['turn_yr']:6.1f}x | " + " ".join(f"{y%100:02d}:{v:4.1f}" for y, v in yr.items()), flush=True)
    return net, gross, info

grids = {
 "main": [dict(k=k, hold=h, reb="d", mode="eq", buf=0, hedge=False) for k in [25, 50, 100] for h in [1, 3, 5, 10]] +
         [dict(k=k, hold=1, reb="w", mode="eq", buf=0, hedge=False) for k in [25, 50, 100]],
 "fine": [dict(k=k, hold=h, reb="d", mode=md, buf=b, hedge=hg) for k in [20, 30, 40] for h in [5, 10] for md in ["eq", "rank"] for b in [0, 20, 50] for hg in [False, True]],
 "legs": [dict(k=k, hold=h, reb="d", mode="eq", buf=0, hedge=hg) for k in [25] for h in [5, 10] for hg in [False, True]],
 "buf": [dict(k=k, hold=h, reb="d", mode="eq", buf=b, hedge=True) for k in [30, 40, 50, 60] for h in [3, 5, 10] for b in [10, 20, 30, 50, 80]],
 "iv": [dict(k=k, hold=h, reb="d", mode="iv", buf=b, hedge=True) for k in [30, 40, 60] for h in [5, 10] for b in [20, 50]],
 "final2": [dict(k=25, hold=5, reb="d", mode="eq", buf=0, hedge=True), dict(k=30, hold=10, reb="d", mode="eq", buf=10, hedge=True),
            dict(k=40, hold=5, reb="d", mode="eq", buf=20, hedge=True), dict(k=25, hold=5, reb="d", mode="eq", buf=0, hedge=False)],
 "final": [dict(k=30, hold=10, reb="d", mode="eq", buf=10, hedge=True), dict(k=25, hold=5, reb="d", mode="eq", buf=0, hedge=False),
           dict(k=25, hold=5, reb="d", mode="eq", buf=0, hedge=True), dict(k=25, hold=1, reb="d", mode="eq", buf=0, hedge=False)],
 "best": [dict(k=40, hold=5, reb="d", mode="eq", buf=20, hedge=True), dict(k=40, hold=5, reb="d", mode="eq", buf=20, hedge=False),
          dict(k=40, hold=5, reb="d", mode="eq", buf=0, hedge=True), dict(k=40, hold=10, reb="d", mode="eq", buf=50, hedge=True),
          dict(k=40, hold=5, reb="d", mode="rank", buf=20, hedge=True)],
}
def legs(name, Wt):
    for nm, Wl in [("LONG ", np.clip(Wt, 0, None)), ("SHORT", np.clip(Wt, None, 0))]:
        net, gross, info = backtest_W(P, F, Wl, cost_bps=args.cost, borrow_bps=args.borrow)
        yr = gross.groupby(gross.index.year).apply(sh)
        print(f"   {nm} leg {name:30s} gross Sh {sh(gross):5.2f} ret {gross.mean()*252*100:6.1f}% vol {gross.std()*np.sqrt(252)*100:5.1f}% | " + " ".join(f"{y%100:02d}:{v:4.1f}" for y, v in yr.items()), flush=True)

res = []
for g in grids[args.grid]:
    Wt = build_W(**g)
    name = f"k{g['k']} h{g['hold']} {g['reb']} {g['mode']} buf{g['buf']} {'H' if g['hedge'] else ''}"
    net, gross, info = evaluate(name, Wt)
    if args.grid == "legs": legs(name, Wt)
    res.append(dict(**g, sh_net=sh(net), sh_gross=sh(gross), ret=net.mean() * 252, vol=net.std() * np.sqrt(252), turn=info["turn_yr"]))
    if args.grid in ("final", "final2"):
        tagn = name.replace(" ", "_") + (f"_sm{args.smooth}" if args.smooth > 1 else "")
        net.to_frame("ret").to_csv(f"/home/trumpet/strat20/agents/ml/logs/net_{args.pred.replace(',', '+')}_{tagn}.csv", index_label="date")
        legs(name, Wt)
    if args.wsave and len(res) == 1:
        np.save(f"{OUT}/W_{args.wsave}.npy", Wt.astype(np.float32)); net.to_csv(f"{OUT}/net_{args.wsave}.csv"); gross.to_csv(f"{OUT}/gross_{args.wsave}.csv")
        if args.grid in ("best", "legs"): legs(name, Wt)
df = pd.DataFrame(res).sort_values("sh_net", ascending=False)
pd.set_option("display.width", 250); print(df.head(15).to_string())
