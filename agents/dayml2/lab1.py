"""Construction lab on the SAVED quarterly baseline predictions (no refit): k, weighting, long-only, rank tiers,
dispersion skip, exit swaps (30/90-bin exits with the 60d/120d models), 60d/120d ensemble.  All at $3/order on $0.4M."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml2")
from d2lib import *
t0 = time.time(); tks = rlib.universe(1500); D = build(tks, verbose=False); log(f"lab1: rows {len(D)} {time.time()-t0:.0f}s")
rows = []
def add(r, rn, label):
    d = ev(r, rn, label); rows.append(d)
preds = {y: load_preds(D, y) for y in ["y_60d", "y_120d"]}
for y in ["y_60d", "y_120d"]:
    pr = preds[y]
    for k in [4, 6, 8, 10]:
        for w in ["eq", "pred"]:
            r, rn = book(D, pr, y, k=k, weight=w); add(r, rn, f"{y} ls k{k} {w}")
    for k in [4, 6, 8, 12]:
        r, rn = book(D, pr, y, k=k, mode="lo", weight="eq"); add(r, rn, f"{y} LO k{k} eq")
    rk = D["rank"].to_numpy()
    for lo, hi in [(1, 600), (601, 1500), (301, 1500), (601, 1000), (1001, 1500), (1, 300), (301, 600)]:
        r, rn = book(D, pr, y, k=6, weight="eq", mask=(rk >= lo) & (rk <= hi)); add(r, rn, f"{y} ls k6 eq ranks {lo}-{hi}")
    # dispersion skip: cross-sectional std of the prediction (and top/bottom spread) vs its trailing median (lagged)
    ps = pd.Series(pr, index=D.index); g = ps.groupby(D.date)
    sd = g.std(); spread = g.apply(lambda s: s.nlargest(6).mean() - s.nsmallest(6).mean()) if False else None
    for win in [60, 120, 250]:
        thr = sd.rolling(win, min_periods=20).median().shift(1); skip = (sd < thr)
        r, rn = book(D, pr, y, k=6, weight="eq", skip=skip); add(r, rn, f"{y} ls k6 eq skip sd<med{win} ({skip.mean()*100:.0f}% days flat)")
    # absolute dispersion filter: top decile / quartile of the day's dispersion only (lagged ranking)
    for q in [0.25, 0.5]:
        thr = sd.rolling(250, min_periods=60).quantile(q).shift(1); skip = (sd < thr)
        r, rn = book(D, pr, y, k=6, weight="eq", skip=skip); add(r, rn, f"{y} ls k6 eq skip sd<q{q}/250d ({skip.mean()*100:.0f}% flat)")
    # exit swaps: same predictions, other exits
    for y2 in ["y_30d", "y_90d", "y_60d", "y_120d"]:
        if y2 == y: continue
        r, rn = book(D, pr, y2, k=6, weight="eq"); add(r, rn, f"{y}-model traded to {y2} ls k6 eq")
# ensemble of the two models (average of per-day pct ranks)
def xr(p): s = pd.Series(p, index=D.index); return (s.groupby(D.date).rank(pct=True) - 0.5).to_numpy()
pe = (xr(preds["y_60d"]) + xr(preds["y_120d"])) / 2
for y in ["y_60d", "y_90d", "y_120d"]:
    for k in [6, 8]:
        r, rn = book(D, pe, y, k=k, weight="eq"); add(r, rn, f"ens60+120 traded to {y} ls k{k} eq")
# where do the k6 positions sit by rank tier, and the yearly picture of a rank-600+ book
pr = preds["y_60d"]; ps = pd.Series(pr, index=D.index); okm = ps.notna()
g = D.loc[okm, ["date", "rank"]].copy(); g["p"] = ps[okm].values; g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size")
sel = g[(g.rk <= 6) | (g.rk > g.n - 6)]
log("k6 positions by rank tier [<=300, 300-600, 600-1000, 1000-1500]: " + str(np.round(np.histogram(sel["rank"], bins=[0, 300, 600, 1000, 1500])[0] / len(sel), 3)))
for yr_, s in sel.groupby(sel.date.dt.year): log(f"  {yr_}: median rank of the k6 positions {s['rank'].median():.0f}, share > 600: {(s['rank'] > 600).mean():.2f}")
T = pd.DataFrame(rows); T.to_csv(f"{HERE}/lab1.csv", index=False)
pd.set_option("display.width", 250); log("\n" + T.sort_values("fo_net", ascending=False).head(25).to_string(index=False))
log(f"LAB1 DONE {time.time()-t0:.0f}s")
