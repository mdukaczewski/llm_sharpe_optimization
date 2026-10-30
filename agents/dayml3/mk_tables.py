#!/usr/bin/env python3
"""markdown tables for REPORT.md from the out/*.csv of the steps"""
import sys, os, glob; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
import pandas as pd, numpy as np
OUT = "/home/user/llm_sharpe_optimization/agents/dayml3/out"
def f2(x): return f"{x:.2f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else str(x)
def md(df, cols, hdr=None):
    hdr = hdr or cols; s = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows(): s += "| " + " | ".join(f2(r[c]) for c in cols) + " |\n"
    return s
def step1():
    T = pd.read_csv(f"{OUT}/step1.csv"); T["f"] = T.f.astype(str)
    rows = []
    for lab, g in T.groupby("label", sort=False):
        d = {"book": lab}
        for f in ["1.0", "0.5", "0.25", "old", "gross-comm"]:
            x = g[g.f == f]
            if len(x): d[f"Sh f={f}"] = x.sh.iloc[0]; d[f"fo f={f}"] = x.fo.iloc[0]
        x = g[g.f == "0.5"].iloc[0]; d["halves f=0.5"] = f"{x.h1:.2f}/{x.h2:.2f}"; d["mdd f=0.5"] = f"{x.mdd*100:.0f}%"
        x = g[g.f == "old"].iloc[0]; d["halves old"] = f"{x.h1:.2f}/{x.h2:.2f}"; d["mdd old"] = f"{x.mdd*100:.0f}%"; rows.append(d)
    R = pd.DataFrame(rows); cols = [c for c in R.columns]
    return md(R, cols)
def step1_tiers():
    T = pd.read_csv(f"{OUT}/step1_tiers.csv"); T = T[T.book.isin(["ens_e30_k8_u1500", "ens_e30_k6_u1500"])]
    T["gross_bps"] = T.gross_bps.round(1); T["cost_bps"] = T.cost_bps.round(1); T["net_bps"] = T.net_bps.round(1); T["hs_in"] = T.hs_in.round(1); T["hs_out"] = T.hs_out.round(1)
    return md(T, ["book", "f", "tier", "share", "hs_in", "hs_out", "gross_bps", "cost_bps", "net_bps"], ["book", "f", "tier", "share of positions", "median hs 09:33 (bps)", "median hs 10:01", "gross bps/trade", "spread cost bps/trade", "net bps/trade"])
def step2():
    T = pd.read_csv(f"{OUT}/step2.csv"); B = pd.read_csv(f"{OUT}/step2b.csv")
    cap = T[T.kind == "cap"].pivot_table(index=["f", "cap"], columns="k", values="fo").reset_index(); cap.columns = [str(c) for c in cap.columns]
    s = "Cap on f x (hs0931 + hs1001): fold-OOS Sharpe at $0.4M by k (rows: f, cap in bps):\n\n" + md(cap, list(cap.columns))
    el = T[T.kind == "cap"].groupby(["f", "cap"]).elig.first().reset_index(); s += "\nShare of (date, name) rows eligible: " + ", ".join(f"f={r.f} cap{int(r.cap)}: {r.elig:.2f}" for r in el.itertuples()) + "\n\n"
    ns = T[T.kind == "netscore"][["label", "sh", "fo", "ret", "mdd", "h1", "h2", "share_le300", "share_gt600"]].copy(); ns["kind"] = "linear"
    nb = B[["label", "sh", "fo", "ret", "mdd", "h1", "h2", "share_le300", "share_gt600"]].copy(); nb["kind"] = "bucket"
    N = pd.concat([ns, nb]); N["ret"] = (N.ret * 100).round(1); N["mdd"] = (N.mdd * 100).round(0)
    s += "Net score = E[y_30d | rank] - f x round-trip cost (linear or bucket calibration), top-k of each side:\n\n" + md(N, ["label", "kind", "sh", "fo", "ret", "mdd", "h1", "h2", "share_le300", "share_gt600"], ["book", "calibration", "Sh", "fold-OOS", "ret %", "maxDD %", "h1", "h2", "share ranks<=300", "share >600"])
    return s
def step3(fn="step3_Y.csv", extra=None):
    T = pd.read_csv(f"{OUT}/{fn}")
    if extra: T = pd.concat([T, pd.read_csv(f"{OUT}/{extra}")])
    T["f"] = T.f.astype(str); rows = []
    for (cfg, uni, k), g in T.groupby(["cfg", "uni", "k"], sort=False):
        d = {"model": cfg, "universe": uni, "k": k}
        for f in ["1.0", "0.5", "0.25", "old"]:
            x = g[g.f == f]
            if len(x): d[f"Sh f={f}"] = x.sh.iloc[0]; d[f"fo f={f}"] = x.fo.iloc[0]
        x = g[g.f == "0.5"].iloc[0]; d["halves f=0.5"] = f"{x.h1:.2f}/{x.h2:.2f}"; d["mdd f=0.5"] = f"{x.mdd*100:.0f}%"; rows.append(d)
    R = pd.DataFrame(rows); return md(R, list(R.columns))
if __name__ == "__main__":
    what = sys.argv[1]; print(globals()[what](*sys.argv[2:]))
