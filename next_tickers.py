import numpy as np, pandas as pd, sys, os
z = np.load("/home/trumpet/strat20/data/panel_liquid.npz", allow_pickle=True)
close = z["close"].astype(np.float64); vol = z["vol"].astype(np.float64); dates = pd.to_datetime(z["dates"])
dv = pd.DataFrame(close * vol, index=dates, columns=z["tickers"]).loc["2016-10-28":"2026-10-27"]
ok = np.isin(z["ttype"], ["CS", "ADRC"])
have = {f.split(".")[0] for f in os.listdir("/home/trumpet/strat20/data/minute_npz")} | {f.split("_")[0] for f in os.listdir("/home/trumpet/strat20/data/minute")}
# median daily dollar volume over the period, require >= 2000 bars (present most of the decade)
med = dv.median(); nb = dv.notna().sum()
cand = med[ok & (nb.values >= 2000)].sort_values(ascending=False)
cand = [t for t in cand.index if t not in have and "." not in t][:int(sys.argv[1])]
print(",".join(cand))
