#!/usr/bin/env python3
"""xsrule3 rule books (cAx = 0.5 z30 - z60 - 0.5 z5l + 0.3 zon_prev - rank(vlast_x), LO top-10 with cAx > 1.5, MOC entry + MOO exit)
re-done with the 15:49:00 cutoff (cut 378): z60 / z5l / vlast_x / rv20 from the cut frame, z30 / zon_prev unchanged.
rank(vlast_x) over universe(1500) per day as in agents/xsrule.  Uses agents/xsrule3/x3lib unchanged (its frame is read, not rewritten)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule3")
from lib6 import log as log6, OUT, rlib, np, pd, sh
import x3lib
from x3lib import positions, net, rep, cax, fold_oos
x3lib.log = log6
D = x3lib.load_frame(600)
X = pd.read_parquet(f"{OUT}/cut378.parquet", columns=["date", "tk", "z60", "z5l", "vlast_x", "rv20"])
X = X[X.tk.map(rlib.RANK).fillna(9999) <= 1500]
X = X.merge(pd.read_parquet(x3lib.load_frame.__globals__["OUT"] + f"/D_u600_{rlib.DEV_END}.parquet", columns=["date"]).drop_duplicates(), on="date")   # same dates (half days dropped)
X["r_vlast_x"] = (X.groupby("date").vlast_x.rank(pct=True) - 0.5).astype(np.float32)
X = X.rename(columns={c: c + "_49" for c in ["z60", "z5l", "r_vlast_x", "rv20"]})
D49 = D.merge(X[["date", "tk", "z60_49", "z5l_49", "r_vlast_x_49", "rv20_49"]], on=["date", "tk"], how="left")
D49 = rlib.clean(D49, ["z60_49", "z5l_49"])
D49["z60"] = D49.z60_49; D49["z5l"] = D49.z5l_49; D49["r_vlast_x"] = D49.r_vlast_x_49; D49["rv20"] = D49.rv20_49
REF = pd.read_csv("/home/user/llm_sharpe_optimization/agents/xsrule3/cands/xsr3_cAx_u50_moo_lo10.csv", index_col=0, parse_dates=True).iloc[:, 0]
rows = []
for nm, F in [("orig (15:59 + official-close rv20)", D), ("cut 15:49", D49)]:
    s0 = cax(F)
    for u in [50, 150]:
        sub = F["rank"] <= u
        P = positions(F[sub], s0[sub], k=10, mode="lo", smin=1.5)
        for c in [0.3, 0.1, 1.0]:
            r = net(P, exit="oo", c=c); d = rep(r, f"xsr {nm} u{u} lo10 c{c}", 10, p=False)
            rc = r - x3lib.comm_bps(10) / 1e4; yr = rc.groupby(rc.index.year).apply(sh)
            rows.append(dict(info=nm, u=u, c=c, sh=round(d["sh"], 2), fold=round(d["oos"], 2), h1=round(d["h1"], 2), h2=round(d["h2"], 2), mdd=round(d["mdd"] * 100, 1),
                             picks_day=round(len(P) / P.date.nunique(), 1), gross_bps=round((P.y_oo).mean() * 1e4, 2), corr_saved_u50=round(r.corr(REF.reindex(r.index)), 2),
                             yrs=" ".join(f"{v:+.1f}" for v in yr.values)))
        if nm.startswith("cut"):
            r = net(P, exit="oo", c=0.3); globals().setdefault("SAVE", {})[u] = r
T = pd.DataFrame(rows); pd.set_option("display.width", 250)
log6("--- xsrule3 cAx LO10 (cAx>1.5) MOC+MOO at the 15:49 cutoff vs the saved 15:59 construction ($0.4M, $3/order)\n" + T.to_string(index=False))
T.to_csv(f"{OUT}/xs6.csv", index=False)
if "save=1" in sys.argv:
    for u, r in SAVE.items():
        rlib.save_cand("onml6", f"onml6_xsr_cAx49_u{u}_moo_lo10", r, f"xsrule3 cAx rule (fixed weights) with z60 / z5l / rank(vlast_x) / rv20 rebuilt to the 15:49:00 cutoff (cut 378: bins to the 15:48 bar), LO top10 cAx>1.5 on the {u} most liquid names, MOC entry + MOO exit, auction cost c=0.3, no borrow; net of cost, BEFORE commission (lo10)")
