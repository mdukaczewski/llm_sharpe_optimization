#!/usr/bin/env python3
"""Step 3: one netted overnight book.  (a) rank-average of p40 and p26 as one prediction; (b) the 50/50 mix of the p40 and p26
cost-aware books with positions NETTED per name (weights 0.5 wA + 0.5 wB, auction cost at the netted order size |w| x $0.4M,
one commission per distinct name); (c) (a) + the xsrule composite cAx (z-scored within day on the 150 most liquid names,
weight w x 0.29 = w x the sd of a percentile rank) as an extra ranking signal.  Sweep k in {6,8,10,12}, lam in {2,3,4}.
usage: python3 step3.py ENTRY EXIT [c=0.3] [f=1]"""
from lib4 import *
ENTRY, EXIT = sys.argv[1], sys.argv[2]; kw = dict(a.split("=") for a in sys.argv[3:] if "=" in a)
C = float(kw.get("c", "0.3")); F = float(kw.get("f", "1"))
G = load_G()
def prank(s, d): return (s.groupby(d).rank(pct=True) - 0.5).astype(np.float32)
G["pR"] = prank((G.p40 + G.p26) / 2, G.date)
for w in [0.5, 1.0]: G[f"pRX{w:g}"] = prank((G.p40 + G.p26) / 2 + w * 0.29 * G.zcAx.fillna(0), G.date)
rows = []
def rec(nm, r, k, L=None, npos=None):
    m = metrics(r, k); row = mrow(nm, m)
    if L is not None: row["t1_pct"] = round((L.tier == 0).mean() * 100, 1); row["gross_bps"] = round((np.sign(L.w) * L.y).mean() * 1e4, 2)
    if npos is not None: row["npos"] = round(npos, 1)
    rows.append(row); log(str({k_: v for k_, v in row.items() if k_ != "yrs"})); return m
Hs = {}
for pc in ["p40", "p26", "pR", "pRX0.5", "pRX1"]:
    H = prep(G, pc); Hs[pc] = H
    for k in [6, 8, 10, 12]:
        for lam in [2.0, 3.0, 4.0]:
            if pc in ("p40", "p26") and lam != 3.0: continue
            r, L = route_book(H, k, lam, ENTRY, EXIT, C, F, ret_pos=True); rec(f"{pc} {ENTRY}->{EXIT} k{k} lam{lam:g}", r, k, L)

# (b) netted 50/50 mix of the p40 and p26 books (fixed routes only: the netted order size changes the auction cost)
def netted(k, lam, wA=0.5):
    HA, HB = Hs["p40"], Hs["p26"]
    _, LA = route_book(HA, k, lam, ENTRY, EXIT, C, F, ret_pos=True); _, LB = route_book(HB, k, lam, ENTRY, EXIT, C, F, ret_pos=True)
    LA = LA[["date", "tk", "w", "exit", "entry"]].assign(w=LA.w * wA); LB = LB[["date", "tk", "w", "exit", "entry"]].assign(w=LB.w * (1 - wA))
    N = pd.concat([LA, LB]).groupby(["date", "tk"]).agg(w=("w", "sum"), exit=("exit", "first"), entry=("entry", "first")).reset_index()
    N = N[N.w.abs() > 1e-12].merge(HA[["date", "tk", "tier", "y_oo", "y_1001", "y_0935", "y_1031", "y_split", "rv20", "dv0", "dvc", "ldv20", "hs1559", "n_hs1001", "hs1001", "n_hs0935", "hs0935"]], on=["date", "tk"], how="left")
    order = N.w.abs().values * CAP
    def auc(o, side): return auction(N, o, C, side).values
    ent = {"moc": auc(order, "close"), "x1559": F * N.hs1559.values, "split": 0.5 * auc(order / 2, "close") + 0.5 * F * N.hs1559.values}
    ex = {"moo": auc(order, "open"), "x1001": F * N.n_hs1001.fillna(N.hs1001).values, "x1031": F * N.n_hs1001.fillna(N.hs1001).values,
          "x0935": F * N.n_hs0935.fillna(N.hs0935).values, "split": 0.5 * auc(order / 2, "open") + 0.5 * F * N.n_hs1001.fillna(N.hs1001).values}
    ycol = {"moo": "y_oo", "x1001": "y_1001", "x1031": "y_1031", "x0935": "y_0935", "split": "y_split"}
    N["y"] = [N.at[i, ycol[e]] for i, e in zip(N.index, N.exit)] if N.exit.nunique() > 1 else N[ycol[N.exit.iloc[0]]]
    ce = np.choose(N.entry.map({e: i for i, e in enumerate(ent)}).values, list(ent.values()))
    cx = np.choose(N.exit.map({e: i for i, e in enumerate(ex)}).values, list(ex.values()))
    N["c"] = ce + cx; N["xo"] = (N.exit == "split").astype(float) + (N.entry == "split").astype(float)
    N["g"] = N.w * N.y; N["cost"] = N.w.abs() * N.c / 1e4 + N.w.clip(upper=0).abs() * 0.01 / 252 + N.xo * ORDER_USD / CAP; N["net"] = N.g - N.cost
    D = N.groupby("date").agg(g=("g", "sum"), cost=("cost", "sum"), npos=("w", "size"), gl=("w", lambda s: s[s > 0].sum()))
    K = int(np.ceil(D.npos.mean() / 2))
    # exact commission: the coordinator subtracts 2K positions x 2 orders; the true count is npos -> adjust inside the series
    r = (D.g - D.cost - (D.npos - 2 * K) * 2 * ORDER_USD / CAP).loc[rlib.START:rlib.DEV_END]
    return r, K, N, D
for k in [6, 8, 10, 12]:
    for lam in [2.0, 3.0, 4.0]:
        r, K, N, D = netted(k, lam)
        rec(f"net50/50 p40+p26 {ENTRY}->{EXIT} k{k}+k{k} lam{lam:g} (named k{K})", r, K, N.assign(y=N.y), npos=D.npos.mean())
        if k == 6 and lam == 3.0:
            log(f"  netting k6 lam3: mean distinct names/day {D.npos.mean():.1f} of 24 slots; gross long weight {D.gl.mean():.2f}; "
                f"names in both books same side {(N.w.abs() > 0.5 / k + 1e-9).mean()*100:.1f}% of positions")
            N.to_parquet(f"{OUT}/step3_net_k6_{ENTRY}_{EXIT}.parquet")
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/step3_{ENTRY}_{EXIT}_c{C:g}_f{F:g}.csv", index=False)
tlog(f"STEP3 blends {ENTRY}->{EXIT} c{C:g} f{F:g} ($0.4M commission)", T.drop(columns=["yrs"]))
log("STEP3 DONE")
