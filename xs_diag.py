import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
from ilib import *; import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
tks = sys.argv[1].split(","); k = 10
def zsd(x, w=20):
    s = pd.Series(x); return (s / s.rolling(w, min_periods=10).std().shift(1)).values
S, R = {}, {}
for tk in tks:
    M = Min(tk); z60 = np.nan_to_num(zsd(M.r(329, 389))); z30 = np.nan_to_num(zsd(M.r(-1, 29)))
    s = 0.5 * z30 - z60; s[M.half] = np.nan
    ron = np.r_[M.c[1:, 1] / M.close[:-1] - 1, np.nan]; ron[np.abs(ron) > 0.5] = np.nan
    S[tk] = pd.Series(s, M.dates); R[tk] = pd.Series(ron, M.dates)
S = pd.DataFrame(S).loc[START:END]; R = pd.DataFrame(R).reindex(S.index)
W = pd.DataFrame(0.0, index=S.index, columns=S.columns)
for d, row in S.iterrows():
    o = row.dropna().sort_values()
    if len(o) < k + 2: continue
    sel = o.index[-k:]; W.loc[d, sel] = np.where(o[sel] > 0, 1.0 / k, 0.0)
contrib = (W * R.fillna(0)).sum() * 252 / len(S) * 100
days = (W > 0).sum()
df = pd.DataFrame({"ret_contrib_pct_yr": contrib.round(2), "days_held": days, "sh_when_held": [sh((W[c] * R[c])[W[c] > 0] * k) for c in S.columns]}, index=S.columns).sort_values("ret_contrib_pct_yr", ascending=False)
print(df.head(12).to_string()); print("..."); print(df.tail(5).to_string())
print("names ever picked:", (days > 0).sum(), " top5 share of positive contrib:", round(df.ret_contrib_pct_yr.head(5).sum() / df.ret_contrib_pct_yr[df.ret_contrib_pct_yr > 0].sum(), 2))
# avg s of picks and % of days with full 10 picks
print("avg picks/day:", (W > 0).sum(axis=1).mean())
