import numpy as np, pandas as pd
from bt import *
P = Panel()
is_cs = np.isin(P.ttype, ["CS", "ADRC", ""])
adv20 = rolling_mean(P.dollar_vol, 20, 15)
m = is_cs[None, :] & (P.close >= 5) & (np.nan_to_num(adv20) >= 5e6)
# previous-day mask (positions held were chosen when liquid)
mprev = np.vstack([np.zeros((1, P.N), bool), m[:-1]])
r = np.where(mprev, P.ret, np.nan)
big = np.argwhere(np.abs(np.nan_to_num(r)) > 1.0)
print("count |ret|>100% in liquid universe:", len(big))
rows = []
for i, j in big:
    rows.append((P.dates[i].date(), P.tickers[j], P.ttype[j], P.close_raw[i-1, j], P.close_raw[i, j], P.close[i-1,j], P.close[i,j], r[i, j], P.vol[i-1,j], P.vol[i,j]))
df = pd.DataFrame(rows, columns=["date","tk","type","craw_prev","craw","cadj_prev","cadj","ret","vol_prev","vol"])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(df.sort_values("ret", ascending=False).head(60))
print(df.sort_values("ret").head(20))
# big negative
neg = np.argwhere(np.nan_to_num(r) < -0.6)
print("count ret<-60%:", len(neg))
rows = []
for i, j in neg:
    rows.append((P.dates[i].date(), P.tickers[j], P.ttype[j], P.close_raw[i-1, j], P.close_raw[i, j], r[i, j], P.vol[i-1,j], P.vol[i,j]))
print(pd.DataFrame(rows, columns=["date","tk","type","craw_prev","craw","ret","vol_prev","vol"]).sort_values("ret").head(60))
