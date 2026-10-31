"""the dev onml3 frame (built 2026-10-30 10:34) carries hs0931/hs1001/hs1559 from data/spread_model_v1.pkl (the model at that time);
recompute those columns of the extended frame with v1 and drop hs0935 (added by the 15:19 refit), so dev rows are identical."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/llm_sharpe_optimization"); import liq, rlib
n = sys.argv[1]
D = pd.read_parquet(n); M = pickle.load(open(f"{rlib.DATA}/spread_model_v1.pkl", "rb"))
parts = []
for tk in D.tk.unique():
    d = liq.raw_frame(tk); e = pd.DataFrame(index=d.index)
    for t in ["0931", "1001", "1559"]:
        m, fac = M[t]; e[f"hs{t}"] = (fac * np.exp(m.predict(d[liq.FEATS[t]].astype(np.float64)))).astype(np.float32)
    e = e.reset_index(); e["tk"] = tk; parts.append(e)
E = pd.concat(parts)
cols = list(D.columns); D = D.drop(columns=["hs0931", "hs1001", "hs1559", "hs0935"])
D = D.merge(E, on=["date", "tk"], how="left")
D[["hs0931", "hs1001", "hs1559"]] = D[["hs0931", "hs1001", "hs1559"]].replace([np.inf, -np.inf], np.nan).astype(np.float32)
D = D[[c for c in cols if c != "hs0935"]]
D.to_parquet(n); print("patched", len(D), flush=True)
