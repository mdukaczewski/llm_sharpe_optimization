import numpy as np, pandas as pd
z = np.load("/home/trumpet/strat20/data/panel.npz", allow_pickle=True)
close = z["close"]; vol = z["vol"]; adj = z["adj"]
dv = close.astype(np.float64) * vol.astype(np.float64)
adv = pd.DataFrame(dv).rolling(20, min_periods=15).mean().values
keep = (np.nanmax(adv, axis=0) >= 10e6)
ttype = z["ttype"]
keep |= np.isin(ttype, ["ETF"]) & (np.nanmax(adv, axis=0) >= 2e6)
print("keeping", keep.sum(), "of", len(keep))
out = {k: z[k] for k in ["dates"]}
for k in ["tickers", "ttype", "adj", "div", "open", "high", "low", "close", "vw", "vol", "n"]:
    a = z[k]; out[k] = a[:, keep] if a.ndim == 2 else a[keep]
np.savez("/home/trumpet/strat20/data/panel_liquid.npz", **out)
print("saved")
