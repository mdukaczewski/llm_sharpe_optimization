#!/usr/bin/env python3
"""Step 0: build out/G.parquet (see lib4 docstring).  2 processes."""
from lib4 import *
from rlib import Min
from multiprocessing import Pool
HS = ["hs0931", "hs0935", "hs1001", "hs1559"]

def one(tk):
    try:
        M = Min(tk)                                       # default end = DEV_END (no holdout bars)
        fwd = lambda x: np.r_[x[1:], np.nan]
        base = M.close * fwd(M.sf); oo = liq.official_open(tk, M.dates); oo = np.where(np.isfinite(oo), oo, M.open)
        d = pd.DataFrame({"y_oo": fwd(oo) / base - 1, "y_on": fwd(M.c[:, 1]) / base - 1, "y_0935": fwd(M.c[:, 5]) / base - 1,
                          "y_1001": fwd(M.c[:, 31]) / base - 1, "y_1031": fwd(M.c[:, 61]) / base - 1}, index=pd.DatetimeIndex(M.dates))
        d[d.abs() > 0.5] = np.nan
        d["r_moc"] = M.close / M.c[:, 389] - 1
        d["rv20"] = pd.Series(M.cc).rolling(20).std().values * np.sqrt(252)
        e = liq.liq_frame(tk).loc[:rlib.DEV_END].reindex(d.index)
        for c in ["ldv20", "dv0", "dvc"] + HS: d[c] = e[c].values
        for c in ["hs0931", "hs0935", "hs1001"]: d["n_" + c] = np.r_[e[c].values[1:], np.nan]
        d = d.astype(np.float32); d.index.name = "date"; d = d.reset_index(); d["tk"] = tk
        return d
    except Exception as ex:
        print(tk, "skip", ex, flush=True); return None

if __name__ == "__main__":
    t0 = time.time()
    P40 = pd.read_parquet(f"{HERE}/agents/onml3/out/pred_f40_on.parquet", columns=["date", "tk", "p"]).rename(columns={"p": "p40"})
    P26 = pd.read_parquet(f"{HERE}/agents/onml3/out/pred_f26calx_q3.parquet", columns=["date", "tk", "p"]).rename(columns={"p": "p26"})
    P = P40.merge(P26, on=["date", "tk"], how="outer"); P = P[P.date <= rlib.DEV_END]
    tks = sorted(P.tk.unique()); log(f"step0: {len(tks)} names, preds {len(P)} rows")
    parts = []
    with Pool(2) as pool:
        for i, d in enumerate(pool.imap_unordered(one, tks, chunksize=4)):
            if d is not None: parts.append(d)
            if i % 200 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
    X = pd.concat(parts)
    G = P.merge(X, on=["date", "tk"], how="left")
    G["rank"] = G.tk.map(rlib.RANK).fillna(9999).astype(np.int32)
    G["tier"] = np.select([G["rank"] <= 300, G["rank"] <= 600, G["rank"] <= 1500], [0, 1, 2], 3).astype(np.int8)
    # xsrule cAx on the 150 most liquid names, z-scored within day
    XS = pd.read_parquet(f"{HERE}/agents/xsrule3/out/D_u600_2026-04-27.parquet", columns=["date", "tk", "rank", "z30", "z60", "z5l", "zon_prev", "r_vlast_x"])
    XS = XS[XS["rank"] <= 150].copy()
    XS["cAx"] = 0.5 * XS.z30 - XS.z60 - 0.5 * XS.z5l + 0.3 * XS.zon_prev - XS.r_vlast_x
    XS["zcAx"] = XS.groupby("date").cAx.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    G = G.merge(XS[["date", "tk", "cAx", "zcAx"]], on=["date", "tk"], how="left")
    G.to_parquet(f"{OUT}/G.parquet")
    # consistency check against the onml3 frame
    F = pd.read_parquet(f"{HERE}/agents/onml3/out/frame_u2000.parquet", columns=["date", "tk", "y_oo", "y_on", "hs0931", "dv0", "rv20"])
    J = G.merge(F, on=["date", "tk"], suffixes=("", "_f"))
    log(f"step0 done {len(G)} rows {G.tk.nunique()} names {time.time()-t0:.0f}s; vs onml3 frame: rows {len(J)}, "
        + " ".join(f"{c} corr {J[c].corr(J[c+'_f']):.5f} maxdiff {(J[c]-J[c+'_f']).abs().max():.2e}" for c in ["y_oo", "y_on", "hs0931", "dv0", "rv20"]))
    log("step0 NaN share: " + " ".join(f"{c} {G[c].isna().mean():.3f}" for c in ["y_oo", "y_on", "y_0935", "y_1001", "y_1031", "hs0935", "n_hs1001", "zcAx"]))
