#!/usr/bin/env python3
"""onml6 books on the onml4 pipeline (lib4: causal yearly calibration map, cost-aware selection lam, k, MOO exit) with the
onml4x cross adverse selection (+1.85 bps per 15:59:30 cross).
Frame H: agents/onml4/out/G.parquet (targets, lagged liquidity, original predictions p40 / p26) + the cut columns:
  rv20 of the cut (selection AND charge use the vol known at the decision), hs1559 rebuilt from the cut inputs.
Books
  moc     (a) every entry a MOC order decided at 15:49:00 (cut 378) on the 15:49 prediction; exit MOO
  two     (b) stage 1 at 15:49:00: per name route class = cheaper of MOC (c) and the expected cross cost (hs1559 from the
              15:49 inputs + 1.85); the top-k long / short by the 15:49 score; selected names of MOC class are sent MOC.
              stage 2 at 15:59:00: the remaining slots (k - MOC names, per side) are filled with 15:59:30 crosses chosen by
              the 15:59:00 (cut 388) prediction and its own cross cost (hs1559 from inputs to 15:59:00 + 1.85), names
              already held excluded.  Every order uses only information available when it is sent.
  x1559   all entries 15:59:30 crosses on the cut-388 prediction (feasible, reference)
  rte     the onml4x per-name entry route on ONE prediction (infeasible for the MOC leg unless the prediction is cut 378)"""
from lib6 import *
import lib4
from lib4 import causal_cal, prep, route_book, commission, metrics, mrow, tier_table, corr_with, auction, book as lbook, CAP, ORDER_USD
ADV = 1.85
_rc = lib4.route_costs
def route_costs(G, k, c=0.3, f=1.0, ls=True):
    C = _rc(G, k, c, f, ls); C["x1559"] = (C["x1559"][0] + ADV, C["x1559"][1] + ADV); return C
lib4.route_costs = route_costs
lib4.LOG = LOG
REFX = f"{HERE}/agents/onml4x/cands/onml4x_pR_lam3_rte_moo_k6.csv"

def prank(s, d): return (s.groupby(d).rank(pct=True) - 0.5).astype(np.float32)

def load_H(preds):
    """preds: dict column -> parquet path (date, tk, p).  Returns the onml4 G frame + cut columns + prediction columns."""
    G = pd.read_parquet(f"{HERE}/agents/onml4/out/G.parquet"); G = G[G.date <= rlib.DEV_END].reset_index(drop=True)
    for cut, sfx in [(378, "49"), (388, "59")]:
        X = pd.read_parquet(f"{OUT}/cut{cut}.parquet", columns=["date", "tk", "rv20", "hs1559"] + (["cb", "c388", "c389", "c_off"] if cut == 378 else []))
        X = X.rename(columns={"rv20": f"rv20_{sfx}", "hs1559": f"hs1559_{sfx}"}); G = G.merge(X, on=["date", "tk"], how="left")
    for col, path in preds.items():
        P = pd.read_parquet(path, columns=["date", "tk", "p"]).rename(columns={"p": col}); G = G.merge(P, on=["date", "tk"], how="left")
    G["rv20_orig"] = G.rv20
    return G

def use_cut(H, sfx):
    """switch the cost inputs of H to the information of cut sfx ('49' / '59' / 'orig')"""
    H = H.copy()
    if sfx == "orig": H["rv20"] = H.rv20_orig; H["hs1559"] = H.hs1559_orig if "hs1559_orig" in H else H.hs1559
    else: H["rv20"] = H[f"rv20_{sfx}"]; H["hs1559"] = H[f"hs1559_{sfx}"]
    return H

def two_stage(H, p1, p2, k=6, lam=3.0, c=0.3, borrow=0.01, ret_pos=False, stage2="all"):
    """(b): H must carry rv20_49, hs1559_49, hs1559_59 and the prediction columns p1 (cut 378) and p2 (cut 388).
    Rows = names with both predictions and y_oo.  Returns the series net of route cost + borrow, before commission."""
    H = H[H[p1].notna() & H[p2].notna() & H.y_oo.notna()].reset_index(drop=True)
    H["p"] = H[p1]; ey1 = causal_cal(H, "y_oo"); H["p"] = H[p2]; ey2 = causal_cal(H, "y_oo")
    order = CAP / (2 * k)
    Hc = H.copy(); Hc["rv20"] = H.rv20_49
    moc = auction(Hc, order, c, "close").values; moo = auction(Hc, order, c, "open").values
    xe1 = H.hs1559_49.values + ADV; xe2 = H.hs1559_59.values + ADV
    is_moc = moc <= xe1
    ent1 = np.where(is_moc, moc, xe1)
    ok = np.isfinite(moc) & np.isfinite(moo) & np.isfinite(xe1) & np.isfinite(xe2)
    H = H[ok].reset_index(drop=True); ey1 = ey1[ok].values; ey2 = ey2[ok].values; moc = moc[ok]; moo = moo[ok]; xe1 = xe1[ok]; xe2 = xe2[ok]; is_moc = is_moc[ok]; ent1 = ent1[ok]
    d = H.date.values
    sL1 = ey1 - lam * (ent1 + moo) / 1e4; sS1 = -ey1 - lam * (ent1 + moo) / 1e4 - borrow / 252
    df = pd.DataFrame({"date": d, "sL1": sL1, "sS1": sS1})
    df["n"] = df.groupby("date").sL1.transform("size"); valid = (df.n >= k + 5).values
    rL1 = df.groupby("date").sL1.rank(method="first", ascending=False).values; rS1 = df.groupby("date").sS1.rank(method="first", ascending=False).values
    selL1 = valid & (rL1 <= k); selS1 = valid & (rS1 <= k)
    mocL = selL1 & is_moc; mocS = selS1 & is_moc & ~mocL
    held = mocL | mocS
    nL = pd.Series(mocL, index=H.index).groupby(d).transform("sum").values; nS = pd.Series(mocS, index=H.index).groupby(d).transform("sum").values
    # stage 2 (15:59:00): fill k - n slots per side with crosses on the 15:59 prediction
    sL2 = ey2 - lam * (xe2 + moo) / 1e4; sS2 = -ey2 - lam * (xe2 + moo) / 1e4 - borrow / 252
    cand = valid & ~held
    if stage2 == "xclass": cand = cand & ~is_moc
    s2 = pd.DataFrame({"date": d, "a": np.where(cand, sL2, -np.inf), "b": np.where(cand, sS2, -np.inf)})
    rL2 = s2.groupby("date").a.rank(method="first", ascending=False).values; rS2 = s2.groupby("date").b.rank(method="first", ascending=False).values
    xL = cand & (rL2 <= k - nL); xS = cand & (rS2 <= k - nS) & ~xL
    rm = H.r_moc.fillna(0).values
    y_x = (1 + H.y_oo.values) * (1 + rm) - 1
    parts = []
    for m, w, y, cst, ent in [(mocL, 1, H.y_oo.values, moc + moo, "moc"), (mocS, -1, H.y_oo.values, moc + moo, "moc"),
                              (xL, 1, y_x, xe2 + moo, "x1559"), (xS, -1, y_x, xe2 + moo, "x1559")]:
        parts.append(pd.DataFrame({"date": d[m], "tk": H.tk.values[m], "tier": H.tier.values[m], "w": w / k, "y": y[m], "c": cst[m], "entry": ent}, index=H.index[m]))
    L = pd.concat(parts); L["g"] = L.w * L.y; L["cost"] = L.w.abs() * L.c / 1e4 + L.w.clip(upper=0).abs() * borrow / 252; L["net"] = L.g - L.cost
    out = L.groupby("date").agg(g=("g", "sum"), cost=("cost", "sum")); r = (out.g - out.cost).loc[rlib.START:rlib.DEV_END]
    return (r, L) if ret_pos else r

def full_metrics(r, k=6):
    m = metrics(r, k); rn = commission(r, k)
    yr = rn.groupby(rn.index.year).apply(sh)
    m["yrs"] = " ".join(f"{y % 100:02d}:{v:+.1f}" for y, v in yr.items())
    return m

def ref_series():
    return pd.read_csv(REFX, index_col=0, parse_dates=True).iloc[:, 0].loc[:rlib.DEV_END]
