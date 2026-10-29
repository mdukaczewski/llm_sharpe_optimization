"""Combinations on the SAVED baseline predictions: {60d, 120d, ensemble} x exit {30d, 60d} x k {6, 8} x skip rule
{none, pred-sd < trailing median (60/120d), gap-dispersion < trailing median}.  Also the yearly profile of the skip."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml2")
from d2lib import *
t0 = time.time(); tks = rlib.universe(1500); D = build(tks, verbose=False); log(f"lab1b: rows {len(D)} {time.time()-t0:.0f}s")
rows = []
def add(r, rn, label): rows.append(ev(r, rn, label))
def xr(p): s = pd.Series(p, index=D.index); return (s.groupby(D.date).rank(pct=True) - 0.5).to_numpy()
p60 = load_preds(D, "y_60d"); p120 = load_preds(D, "y_120d"); pe = (xr(p60) + xr(p120)) / 2
gd = D.groupby("date")["gap_disp"].first()
def skips(pr):
    sd = pd.Series(pr, index=D.index).groupby(D.date).std(); out = {"none": None}
    for win in [60, 120]: out[f"sd<med{win}"] = sd < sd.rolling(win, min_periods=20).median().shift(1)
    out["gapdisp<med60"] = gd < gd.rolling(60, min_periods=20).median().shift(1)
    out["sd<med60&gap"] = out["sd<med60"] | out["gapdisp<med60"]
    return out
for pname, pr in [("m60", p60), ("m120", p120), ("ens", pe)]:
    sk = skips(pr)
    for y in ["y_30d", "y_60d"]:
        for k in [6, 8]:
            for sname, s in sk.items():
                r, rn = book(D, pr, y, k=k, weight="eq", skip=s); add(r, rn, f"{pname} {y} k{k} skip={sname}" + ("" if s is None else f" ({s.mean()*100:.0f}% flat)"))
# yearly share of flat days for the sd<med60 rule of the ensemble, and its Sharpe on traded vs skipped days
sd = pd.Series(pe, index=D.index).groupby(D.date).std(); s60 = sd < sd.rolling(60, min_periods=20).median().shift(1)
r, rn = book(D, pe, "y_60d", k=6, weight="eq")
log("ens y_60d k6: Sh on would-be-skipped days %.2f vs traded days %.2f; flat share by year: %s" % (rlib.sh(rn[s60.reindex(rn.index).fillna(False)]), rlib.sh(rn[~s60.reindex(rn.index).fillna(False)]), s60.groupby(s60.index.year).mean().round(2).to_dict()))
T = pd.DataFrame(rows); T.to_csv(f"{HERE}/lab1b.csv", index=False)
pd.set_option("display.width", 250); log("\n" + T.sort_values("fo_net", ascending=False).head(30).to_string(index=False))
log(f"LAB1B DONE {time.time()-t0:.0f}s")
