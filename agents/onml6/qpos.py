#!/usr/bin/env python3
"""Exact check of the LOC-at-the-15:49-bid books on the quote-panel days: real 15:49:00 NBBO for their positions (quotes49.py),
exact fill (official close strictly through the real bid / ask) vs the proxy fill, and the conditional overnight returns.
usage: python3 qpos.py <Y|Q> fetch|eval"""
from lib6 import *
import subprocess
V = sys.argv[1]; act = sys.argv[2]
P = pd.concat([pd.read_pickle(f"{OUT}/lmt_pos_{V}.pkl").assign(book="pR"), pd.read_pickle(f"{OUT}/lmt_pos40_{V}.pkl").assign(book="p40")], ignore_index=True)
days = sorted(d for d in pd.read_csv(f"{HERE}/out/quotes_panel.csv", usecols=["date"]).date.unique() if d <= rlib.DEV_END)
P = P[P.date.isin(pd.to_datetime(days))]
jobs = f"{LANE}/quotes/jobs_pos_{V}.csv"; qout = f"{LANE}/quotes/q49_pos.csv"

def evaluate(J, bk):
    lg = J.w > 0; sg = np.sign(J.w)
    J["fill_x"] = np.where(lg, J.c_off < J.bid, J.c_off > J.ask)
    J["hs_real"] = (J.ask - J.bid) / (J.ask + J.bid) * 1e4
    out = dict(n=len(J), days=J.date.nunique(), fill_proxy=J.fill.mean() * 100, fill_exact=J.fill_x.mean() * 100, agree=(J.fill == J.fill_x).mean() * 100,
               gross_filled_proxy=(sg * J.y)[J.fill].mean() * 1e4, gross_filled_exact=(sg * J.y)[J.fill_x].mean() * 1e4,
               gross_unfilled_exact=(sg * J.y)[~J.fill_x].mean() * 1e4, hs_real_mean=J.hs_real.mean(), hs_pred_mean=J.hs1559.mean(),
               hs_real_med=J.hs_real.median(), hs_pred_med=J.hs1559.median())
    log(f"--- qpos {V} {bk}: LOC book positions on the quote-panel days, real 15:49:00 NBBO vs proxy\n  " + "  ".join(f"{a}={b:.2f}" if isinstance(b, float) else f"{a}={b}" for a, b in out.items()))
    def pnl(f): return np.where(f, J.w * J.y - J.w.abs() * J.c / 1e4 - np.where(J.w < 0, J.w.abs(), 0) * 0.01 / 252 - 2 * 3.0 / 400000, 0.0)
    a = pd.Series(pnl(J.fill), index=J.index).groupby(J.date).sum(); b = pd.Series(pnl(J.fill_x), index=J.index).groupby(J.date).sum()
    log(f"  panel-day mean daily PnL (incl. commission) proxy {a.mean()*1e4:.2f} bps vs exact {b.mean()*1e4:.2f} bps (n days {len(a)}); corr {a.corr(b):.2f}")

if act == "fetch":
    P[["tk", "date"]].assign(date=P.date.dt.strftime("%Y-%m-%d")).drop_duplicates().to_csv(jobs, index=False)
    print(len(P), "positions on", P.date.nunique(), "panel days", flush=True)
    subprocess.run([sys.executable, f"{LANE}/quotes49.py", jobs, qout, "8"], check=True)
else:
    Q = pd.read_csv(qout, parse_dates=["date"]).dropna(subset=["bid", "ask"])
    X = pd.read_parquet(f"{OUT}/cut378.parquet", columns=["date", "tk", "cb", "c_off", "hs1559"], filters=[("date", "in", list(P.date.unique()))])
    J0 = P.merge(Q, on=["tk", "date"], how="inner").merge(X, on=["date", "tk"], how="left")
    for bk in ["pR", "p40"]: evaluate(J0[J0.book == bk].reset_index(drop=True), bk)
