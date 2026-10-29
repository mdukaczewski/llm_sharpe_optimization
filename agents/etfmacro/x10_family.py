"""Candidate family for the VIX-ETP overnight carry book, all regime variables lagged one day (known before the decision bar)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
D = load_etf(); idx = D["ret"].index; F = load_fred(idx)
ratio_l = (F.VIX / F.VIX3M).shift(1); vrp_l = (F.VIX / 100 - vlib.rv(D["ret"]["SPY"], 10)).shift(1)   # known at close t-1
wd = pd.Series(idx.weekday, index=idx); notmon = (wd != 4).astype(float)
f95 = (ratio_l < 0.95).astype(float); reg2 = ((ratio_l < 0.95) | (vrp_l > 0.03)).astype(float)
rV = {e: on_ret("VIXY", e) for e in [375, 390]}; rU = {e: on_ret("UVXY", e) for e in [375, 390]}
print("check: VIXY on_ret(390) vs ticker_frame y_on corr", round(rV[390].reindex(idx).corr(tf("VIXY").reindex(idx).y_on), 4), "| UVXY extremes", rU[390].sort_values().tail(3).round(2).to_dict())
res = {}
def add(name, n, p=True):
    res[name] = n
    if p: stats(n, name)
for e in [390, 375]:
    tag = "" if e == 390 else "_1545"
    add(f"vixon_r95{tag}", leg_ret(-0.5 * notmon * f95, rV[e], 1.5, 0.02))
    add(f"vixon_r95vrp{tag}", leg_ret(-0.5 * notmon * reg2, rV[e], 1.5, 0.02))
    add(f"uvxon_r95{tag}", leg_ret(-0.25 * notmon * f95, rU[e], 1.5, 0.05))
    add(f"uvxon_r95vrp{tag}", leg_ret(-0.25 * notmon * reg2, rU[e], 1.5, 0.05))
print("-- unlagged regime for comparison (16:15 VIX close used at the 16:00 decision: NOT allowed)")
stats(leg_ret(-0.5 * notmon * (ratio_l.shift(-1) < 0.95), rV[390], 1.5, 0.02), "vixon_r95 unlagged")
print("-- VIXM close-to-close overlay, causal (decided close t, filled close t+1)")
for wv in [0.25, 0.5]:
    h = bt(pd.DataFrame({"VIXM": wv * notmon * f95}, index=idx), D)["net"]
    add(f"vixon_r95_vixm{int(wv*100)}", res["vixon_r95"] + h)
    h2 = bt(pd.DataFrame({"VIXM": wv * notmon * reg2}, index=idx), D)["net"]
    add(f"vixon_r95vrp_vixm{int(wv*100)}", res["vixon_r95vrp"] + h2)
print("== fold-OOS: whole family, then sub-families")
oos, tab = fold_oos(res, label="all"); print(tab.to_string())
fold_oos({k: v for k, v in res.items() if "1545" not in k and "vixm" not in k}, label="close-entry, no hedge")
fold_oos({k: v for k, v in res.items() if "vixm" not in k}, label="no hedge")
fold_oos({k: v for k, v in res.items() if k.startswith("vixon_r95") and "vrp" not in k}, label="vixon r95 variants")
for k, v in res.items(): fold_oos(v, label=k)
print("== correlations")
for k in ["vixon_r95", "vixon_r95vrp", "uvxon_r95", "vixon_r95_1545", "vixon_r95_vixm50"]:
    print(k, end=" "); corr_report(res[k])
print("corr matrix"); print(pd.DataFrame(res).corr().round(2).to_string())
pd.DataFrame(res).to_csv("/home/user/llm_sharpe_optimization/agents/etfmacro/x10_res.csv")
