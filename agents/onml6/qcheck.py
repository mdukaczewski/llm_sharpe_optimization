#!/usr/bin/env python3
"""15:49:00 NBBO sample (quotes/q49_sample.csv) vs the 15:59:30 panel quote and vs the predicted half-spread from the cut-378
inputs (hs1559 booster on bins to 15:49); bid proxy c378 x (1 - s x hs) vs the real bid; LOC fill (official close < bid) exact vs proxy."""
from lib6 import *
Q = pd.read_csv(f"{LANE}/quotes/q49_sample.csv", parse_dates=["date"]).dropna(subset=["bid", "ask"])
Q = Q[(Q.ask >= Q.bid) & (Q.bid > 0)]
P = pd.read_csv(f"{HERE}/out/quotes_panel.csv", parse_dates=["date"], dtype={"t": str})
P = P[P.t == "1559"][["tk", "date", "bid", "ask"]].rename(columns={"bid": "bid59", "ask": "ask59"})
X = pd.read_parquet(f"{OUT}/cut378.parquet", columns=["date", "tk", "hs1559", "cb", "c_off"], filters=[("date", "in", list(Q.date.unique()))])
X59 = pd.read_parquet(f"{OUT}/cut388.parquet", columns=["date", "tk", "hs1559"], filters=[("date", "in", list(Q.date.unique()))]).rename(columns={"hs1559": "hs59"})
J = Q.merge(X, on=["tk", "date"], how="left").merge(X59, on=["tk", "date"], how="left").merge(P, on=["tk", "date"], how="left")
J["hs"] = (J.ask - J.bid) / (J.ask + J.bid) * 1e4; J["hs_59"] = (J.ask59 - J.bid59) / (J.ask59 + J.bid59) * 1e4
J = J[(J.hs > 0) & (J.hs < 500)].dropna(subset=["hs1559", "cb"])
J["rank"] = J.tk.map(rlib.RANK); J["tier"] = pd.cut(J["rank"], [0, 300, 600, 2000], labels=["1-300", "301-600", "601-2000"])
J["mid"] = (J.ask + J.bid) / 2; J["mid_err"] = (J.cb / J.mid - 1) * 1e4
rows = []
for t, g in list(J.groupby("tier", observed=True)) + [("all", J)]:
    rows.append(dict(tier=t, n=len(g), hs1549_mean=g.hs.mean(), hs1549_med=g.hs.median(), hs155930_mean=g.hs_59.mean(), hs155930_med=g.hs_59.median(),
                     pred49_mean=g.hs1559.mean(), pred49_med=g.hs1559.median(), corr_pred=np.corrcoef(np.log(g.hs), np.log(g.hs1559))[0, 1],
                     c378_vs_mid_abs_med=g.mid_err.abs().median()))
T = pd.DataFrame(rows).round(2); log("--- qcheck: real NBBO half-spread at 15:49:00 vs 15:59:30 (panel) vs predicted (hs1559 booster, inputs to 15:49)\n" + T.to_string(index=False))
# LOC buy at the bid / sell at the ask: exact (real 15:49 bid / ask) vs proxy (c378 x (1 -+ s hs_pred)), strict through the official close
for s in [1.0, 0.5]:
    bL = np.round(J.cb * (1 - s * J.hs1559 / 1e4), 2); aL = np.round(J.cb * (1 + s * J.hs1559 / 1e4), 2)
    fx_b = J.c_off < J.bid; fp_b = J.c_off < bL; fx_a = J.c_off > J.ask; fp_a = J.c_off > aL
    log(f"  LOC fill rate s={s}: buy exact {fx_b.mean()*100:.1f}% proxy {fp_b.mean()*100:.1f}% agree {(fx_b==fp_b).mean()*100:.1f}% | sell exact {fx_a.mean()*100:.1f}% proxy {fp_a.mean()*100:.1f}% agree {(fx_a==fp_a).mean()*100:.1f}%; "
        f"proxy bid - real bid median {np.median((bL / J.bid - 1) * 1e4):+.2f} bps (mean {np.mean((bL / J.bid - 1) * 1e4):+.2f})")
J.to_csv(f"{OUT}/qcheck.csv", index=False)
