import numpy as np, pandas as pd, sys
z = np.load("/home/trumpet/strat20/data/panel_liquid.npz", allow_pickle=True)
close = z["close"].astype(np.float64); vol = z["vol"].astype(np.float64)
adv = pd.DataFrame(close * vol).rolling(20, min_periods=15).mean().values
mx = np.nanmax(np.where(np.isnan(adv), 0, adv), axis=0)
keep = (mx >= 30e6) & np.isin(z["ttype"], ["CS", "ADRC"])
tk = z["tickers"][keep]
open(sys.argv[1], "w").write("\n".join(tk))
print(len(tk))
