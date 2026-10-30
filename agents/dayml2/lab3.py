"""Evaluate the yearly-refit predictions of lab2 (out/pred_<cfg>.npy) under the robust constructions:
own exit and the bin-31 exit, k6/k8 eq, with / without the gap-dispersion skip, plus pairwise ensembles.
usage: python3 lab3.py cfg1 cfg2 ... [--ens a+b,c+d]"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml2")
from d2lib import *
args = sys.argv[1:]; ens = []
if "--ens" in args: i = args.index("--ens"); ens = args[i + 1].split(","); args = args[:i]
CFG_Y = {"base": "y_60d", "t600": "y_60d", "cap31": "y_60d", "cap63": "y_60d", "roll3": "y_60d", "roll4": "y_60d", "hl2": "y_60d", "hl1": "y_60d", "thin": "y_60d",
         "ext2": "y_60d", "ext2on": "y_60d", "new9": "y_60d", "y30": "y_30d", "y90": "y_90d", "y120": "y_120d", "ext2_y90": "y_90d", "ext2_y120": "y_120d",
         "ext2on_y120": "y_120d", "ext2_cap31": "y_60d", "ext2on_cap31": "y_60d", "ext2_hl2": "y_60d", "ext2on_y30": "y_30d", "ext2_y30": "y_30d"}
t0 = time.time(); tks = rlib.universe(1500); D = build(tks, verbose=False); log(f"lab3 {args} ens {ens}: rows {len(D)} {time.time()-t0:.0f}s")
gd = D.groupby("date")["gap_disp"].first(); gskip = gd < gd.rolling(60, min_periods=20).median().shift(1)
def xr(p): s = pd.Series(p, index=D.index); return (s.groupby(D.date).rank(pct=True) - 0.5).to_numpy()
def load(cfg): return np.load(f"{HERE}/out/pred_{cfg}.npy")
rows = []
def run(pname, pr, ys):
    for y in ys:
        for k in [6, 8]:
            for sname, s in [("none", None), ("gapdisp<med60", gskip)]:
                r, rn = book(D, pr, y, k=k, weight="eq", skip=s); d = ev(r, rn, f"{pname} {y} k{k} skip={sname}"); d.update(model=pname, y=y, k=k, skip=sname); rows.append(d)
for cfg in args:
    pr = load(cfg); y0 = CFG_Y[cfg]; run(cfg, pr, sorted({y0, "y_30d", "y_60d"}))
for e in ens:
    parts = e.split("+"); pr = sum(xr(load(c)) for c in parts) / len(parts); run("ens:" + e, pr, ["y_30d", "y_60d"])
T = pd.DataFrame(rows); T.to_csv(f"{HERE}/out/lab3_{'_'.join(args)[:50]}.csv", index=False)
pd.set_option("display.width", 250); log("\n" + T.sort_values("fo_net", ascending=False)[["label", "sh", "fo_spread", "sh_net", "fo_net", "ret", "h1", "h2"]].round(2).head(40).to_string(index=False))
log(f"LAB3 DONE {time.time()-t0:.0f}s")
