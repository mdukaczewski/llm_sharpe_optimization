"""Step 3: do minute-bar close-time features (rlib.ticker_frame) and the onml3 overnight predictions (5-day mean) help
the daily model at the 5-10 day horizon?  Yearly retrain, n_jobs=2, same rows / target as the step-1 models.
usage: python s3_extra.py <target tag e.g. s5> <variants comma list: base,min,minon>"""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dml2")
from dlib import *
MINF = ["z30", "z60", "z30l", "z120", "zoc", "zo15", "zgap", "rngpos", "rng", "vlast_share", "vfirst_share", "vmoc_share", "vday_rel", "zvwap", "cc5", "rv5_20", "zon_prev"]
MIN5 = ["z60", "zoc", "z30l", "vlast_share", "zvwap", "zgap"]          # 5-day means as well

def rank_keepnan(a, m):
    df = pd.DataFrame(np.where(m & np.isfinite(a), a, np.nan)); r = df.rank(axis=1, pct=True).to_numpy()
    cnt = df.notna().sum(axis=1).to_numpy()[:, None]; return (r - 0.5 - 0.5 / np.maximum(cnt, 1)).astype(np.float32)

def minute_mats(P):
    out = {f: np.full((P.T, P.N), np.nan, np.float32) for f in MINF}; n = 0
    for tk in P.tickers:
        p = f"{rlib.DATA}/cache/{tk}.v2.parquet"
        if tk not in rlib.RANK or not os.path.exists(p): continue
        d = pd.read_parquet(p, columns=MINF + ["half"]); d = d[d.index <= DEV_END]
        d = d[~d.index.duplicated()].reindex(P.dates); j = P.ix[tk]
        for f in MINF: out[f][:, j] = d[f].to_numpy()
        n += 1
    log(f"minute_mats: {n} names"); return out

def onml_mat(P):
    a = None
    for nm in ["pred_f40_on", "pred_f26calx_q3"]:
        d = pd.read_parquet(f"{HERE}/agents/onml3/out/{nm}.parquet"); d = d[d.date <= DEV_END]
        d["r"] = d.groupby("date").p.rank(pct=True) - 0.5
        w = d.pivot_table(index="date", columns="tk", values="r").reindex(index=P.dates, columns=P.tickers)
        a = w.to_numpy() if a is None else np.nanmean(np.stack([a, w.to_numpy()]), axis=0)
    return a.astype(np.float32)

if __name__ == "__main__":
    tg = sys.argv[1]; variants = sys.argv[2].split(",")
    P, F, m = setup(); rt, ri = np.nonzero(m); rows = (rt, ri)
    X = np.load(f"{OUT}/X16.npy").astype(np.float32); names = [l.strip() for l in open(f"{OUT}/X_names.txt")]
    extra, en = [], []
    if any(v != "base" for v in variants):
        M = minute_mats(P)
        for f in MINF: extra.append(rank_keepnan(M[f], m)[rt, ri]); en.append("x_" + f)
        for f in MIN5:
            a = pd.DataFrame(np.clip(np.nan_to_num(M[f], nan=0.0) if False else M[f], -5, 5)).rolling(5, min_periods=3).mean().to_numpy()
            extra.append(rank_keepnan(a, m)[rt, ri]); en.append("x5_" + f)
        del M
    if "minon" in variants:
        A = onml_mat(P)
        extra.append(rank_keepnan(A, m)[rt, ri]); en.append("on1")
        a5 = pd.DataFrame(A).rolling(5, min_periods=3).mean().to_numpy(); extra.append(rank_keepnan(a5, m)[rt, ri]); en.append("on5")
        log(f"onml coverage on mask rows {np.isfinite(extra[-2]).mean():.3f}")
    h = int(tg[1:]); skip = 1 if tg[0] == "s" else 0
    y = fwd_target(P, F, m, h, skip); i0 = int(np.searchsorted(P.dates, pd.Timestamp(START)))
    for v in variants:
        nmin = len(MINF) + len(MIN5)
        sel = [] if v == "base" else (list(range(nmin)) if v == "min" else list(range(len(extra))))
        Xv = np.concatenate([X] + [extra[i][:, None] for i in sel], axis=1) if sel else X
        nv = names + [en[i] for i in sel]
        t0 = time.time(); pred, imp = walk_forward(P, m, rows, Xv, nv, y, h + skip, retrain="Y")
        save_pred(pred, f"{tg}_Y_{v}"); ic = rank_ic(pred, y, m, i0, P.T); ic.index = P.dates[i0:]
        log(f"S3 {tg} {v}: K={len(nv)} OOS rank IC {ic.mean():.4f} IR {ic.mean()/ic.std():.3f} | {time.time()-t0:.0f}s | by year " +
            " ".join(f"{y_%100}:{v_:.3f}" for y_, v_ in ic.groupby(ic.index.year).mean().items()))
        log("   top feats " + ", ".join(f"{k}:{v_*100:.1f}%" for k, v_ in imp.head(12).items()))
        xi = imp[[c for c in imp.index if c.startswith(("x_", "x5_", "on"))]]
        if len(xi): log(f"   extra feats total gain share {xi.sum()*100:.1f}%: " + ", ".join(f"{k}:{v_*100:.1f}%" for k, v_ in xi.head(8).items()))
        del Xv
