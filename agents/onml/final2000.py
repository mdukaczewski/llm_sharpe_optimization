#!/usr/bin/env python3
"""2000-name overnight book on an IMPLEMENTABLE feature set (decided at the 15:59 bar, filled at the official close):
drops zmoc (closing print), vmoc_share (closing-auction volume share), z5l, and replaces vlast_share / vday_rel / ldv
(which include the 16:00 auction volume) with ex-auction versions.  Yearly seed-0 check of f26cal vs f26calx, then the
quarterly 3-seed book; predictions -> agents/onml/out/pred_f26calx_u2000_q3.parquet; candidates k10/k15/k20 (eq).
Run with STRAT_END=2026-10-28 (series on dev dates are identical to a dev run; every number printed here is cut at
DEV_CUT=2026-04-27 so the holdout stays unread)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
from rlib import Min
NJ = int(os.environ.get("ONML_NJOBS", "2")); CAP = 400000.0; DEV_CUT = "2026-04-27"; N = int(os.environ.get("ONML_N", "2000"))
XD = f"{rlib.DATA}/cache_x"; os.makedirs(XD, exist_ok=True)
def exauction(tk):
    cp = f"{XD}/{tk}.parquet"; src = f"{rlib.DATA}/minute_npz/{tk}.npz"
    if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(src): return pd.read_parquet(cp)
    M = Min(tk, end="2026-10-28"); v = M.v; vday = v[:, :390].sum(1)
    d = pd.DataFrame({"vlast_x": v[:, 330:390].sum(1) / (vday + 1), "vday_rel_x": vday / (pd.Series(vday).rolling(20).mean().shift(1).values + 1),
                      "ldv_x": np.log(vday * M.close + 1)}, index=M.dates).astype(np.float32); d.index.name = "date"
    d.to_parquet(cp); return d
def add_x(D):
    parts = []
    for tk in D.tk.unique():
        e = exauction(tk).reset_index(); e["tk"] = tk; parts.append(e)
    E = pd.concat(parts); return D.merge(E, on=["date", "tk"], how="left")
def dev(r): return r[r.index <= DEV_CUT]
def line(name, r, k):
    r = dev(r); rn = r - 2 * k * 6 / CAP
    return f"{name:34s} k{k}: spread-only Sh {rlib.sh(r):.2f} foldOOS {rlib.sh(rlib.fold_oos(r, p=False)[0]):.2f} | $0.4M Sh {rlib.sh(rn):.2f} foldOOS {rlib.sh(rlib.fold_oos(rn, p=False)[0]):.2f} | halves {rlib.sh(rn[rn.index < '2021-10-28']):.2f}/{rlib.sh(rn[rn.index >= '2021-10-28']):.2f} mdd {((1+rn).cumprod()/(1+rn).cumprod().cummax()-1).min()*100:.0f}%"
t0 = time.time(); D = load(N); D = add_x(D); D = rlib.clean(D, ["vlast_x", "vday_rel_x", "ldv_x"]); log(f"final2000: {D.tk.nunique()} names, DEV_END={rlib.DEV_END}, {time.time()-t0:.0f}s")
FS = FEATS18 + EXTRA10 + CAL
F26 = [f for f in FS if f not in ("zmoc", "vmoc_share", "z5l")]
F26X = [f if f not in ("vlast_share", "vday_rel", "ldv") else f + "_x" for f in F26]
for name, fs in [("f26cal", F26), ("f26calx", F26X)]:
    p = xrank(D, wf(D, fs, "yr", "Y", seed=0, n_jobs=NJ))
    for k in [10, 20]: log("  " + line(f"u{N} {name} yearly s0", book2(D, p, k=k, p=False), k))
members = []
for s in [0, 1, 2]:
    t1 = time.time(); members.append(xrank(D, wf(D, F26X, "yr", "Q", seed=s, n_jobs=NJ))); log(f"  quarterly seed {s} {time.time()-t1:.0f}s")
p = sum(members) / 3
pd.DataFrame({"date": D.date, "tk": D.tk, "p": p}).dropna().to_parquet(f"{HERE}/agents/onml/out/pred_f26calx_u{N}_q3.parquet")
for k in [10, 15, 20]:
    r = book2(D, p, k=k, p=False); name = f"onml_f26calx_u{N}_q3_k{k}"
    rlib.save_cand("onml", name, r, f"3-seed quarterly LightGBM, implementable feature set (no closing print / auction volume / last-5-min), universe({N}), L/S top{k} eq-weight")
    log("  " + line(f"FINAL {name}", r, k))
log(f"FINAL2000 DONE {time.time()-t0:.0f}s")
