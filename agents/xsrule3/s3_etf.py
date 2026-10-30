#!/usr/bin/env python3
"""Step 4: vixon_r95_ls (agents/etfmacro/final.py) with the VIXY exit at the OFFICIAL OPEN (liq.overnight_targets y_oo) instead of
the 09:31 bar, 1.5 bps per side, borrow 2% on the short leg.  Report only (nothing saved unless materially different)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3lib import *
sys.path.insert(0, f"{ROOT}/agents/etfmacro"); import elib
log("s3_etf start")
E = elib.load_etf(); idx = E["ret"].index; F = elib.load_fred(idx)
ratio_l = (F.VIX / F.VIX3M).shift(1); wd = pd.Series(idx.weekday, index=idx); notfri = (wd != 4).astype(float); f95 = (ratio_l < 0.95).astype(float)
spy = elib.tf("SPY").reindex(idx); s_spy = (0.5 * spy.z30 - spy.z60).clip(-5, 5).fillna(0.0)
def oo_ret(tk):
    t = liq.overnight_targets(tk).y_oo.astype(float); t = t.loc[:DEV_END].copy(); t.loc[t.index >= DEV_END] = np.nan   # never read the holdout open
    return t.reindex(idx)
out = {}
for tk, w, b, tag in [("VIXY", 0.5, 0.02, "vixon"), ("UVXY", 0.25, 0.05, "uvxon")]:
    r_on = elib.on_ret(tk, 390); r_oo = oo_ret(tk)
    j = pd.concat([r_on.rename("on"), r_oo.rename("oo")], axis=1).dropna()
    log(f"{tk}: close->09:31 vs close->open: mean bps {j.on.mean()*1e4:.2f} / {j.oo.mean()*1e4:.2f}  sd {j.on.std()*1e4:.0f} / {j.oo.std()*1e4:.0f}  corr {j.on.corr(j.oo):.4f}  n {len(j)}")
    for ex, rr in [("0931", r_on), ("moo", r_oo)]:
        out[f"{tag}_r95_{ex}"] = elib.leg_ret(-w * notfri * f95, rr, 1.5, b)
        if tk == "VIXY": out[f"vixon_r95_ls_{ex}"] = out[f"vixon_r95_{ex}"] + elib.leg_ret(0.5 * ((ratio_l >= 0.95) & (s_spy < -1)).astype(float), rr, 1.5, 0.0)
ref = pd.read_csv(f"{ROOT}/agents/etfmacro/cands/vixon_r95_ls.csv", index_col=0, parse_dates=True).iloc[:, 0].loc[:DEV_END]
log(f"saved vixon_r95_ls (cut at DEV_END): Sh {sh(ref):.3f}; corr with my 0931 rebuild {ref.corr(out['vixon_r95_ls_0931'].reindex(ref.index)):.4f}")
for k, v in out.items():
    s = stats(v, k, p=False); oos, tab = fold_oos(v, p=False)
    log(f"{k:22s} Sh {s['sharpe']:.3f} OOS {sh(oos):.3f} halves {s['h1']:.2f}/{s['h2']:.2f} mdd {s['mdd']*100:.1f}% ret {s['ret']*100:.1f}%")
log("s3_etf done")
