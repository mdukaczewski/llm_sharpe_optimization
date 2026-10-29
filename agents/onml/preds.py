#!/usr/bin/env python3
"""Save the 3-seed f28cal quarterly overnight predictions (full range, STRAT_END from env) to
agents/onml/out/pred_f28cal_q3.parquet and write k10/k15/k20 eq/pred books as candidates
(fixed per-order commissions favour fewer, larger positions)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
NJ = int(os.environ.get("ONML_NJOBS", "2")); F28 = FEATS18 + EXTRA10; FS = F28 + CAL
t0 = time.time(); D = load(1500); log(f"preds: {D.tk.nunique()} names, DEV_END={rlib.DEV_END}, n_jobs={NJ}")
members = []
for s in [0, 1, 2]:
    t1 = time.time(); members.append(xrank(D, wf(D, FS, "yr", "Q", seed=s, n_jobs=NJ))); log(f"  seed {s} {time.time()-t1:.0f}s")
p = sum(members) / 3
pd.DataFrame({"date": D.date, "tk": D.tk, "p": p}).dropna().to_parquet(f"{HERE}/agents/onml/out/pred_f28cal_q3.parquet")
for k in [10, 15, 20]:
    for w in ["eq", "pred"]:
        r = book2(D, p, k=k, weight=w, p=False); name = f"onml_f28cal_q3_k{k}{'p' if w == 'pred' else ''}"
        if name != "onml_f28cal_q3_k20": rlib.save_cand("onml", name, r, f"3-seed f28cal quarterly, L/S top{k} {w}-weighted")
        log(report_line(f"BOOK {name}", r))
log(f"preds done {time.time()-t0:.0f}s")
