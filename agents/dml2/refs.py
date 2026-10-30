"""reference series for correlations (all exit-dated)."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import rlib
H = "/home/user/llm_sharpe_optimization"
def _r(p): return pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0].loc[rlib.START:rlib.DEV_END]
def refs():
    ml = _r(f"{H}/agents/ml/logs/net_lgb1n+lgb5n+lgb10n_k25_h5_d_eq_buf0_H_sm5.csv")
    mom = _r(f"{H}/logs/mom_grid/mom_E4_top25.csv")
    on = pd.read_csv(f"{H}/agents/onml4/cands/onml4_pR_lam3_rte_rtx_k6.csv", index_col=0, parse_dates=True).iloc[:, 0]
    on = on.shift(1).loc[rlib.START:rlib.DEV_END]          # decision-dated -> exit-dated (one trading day later)
    return {"ml_k25_h5": ml, "mom_E4_top25": mom, "onml4_rtx_k6(+1d)": on}
def corrs(r):
    out = {}
    for k, s in refs().items():
        j = pd.concat([r, s], axis=1, join="inner").fillna(0.0); out[k] = round(j.corr().iloc[0, 1], 3)
    return out
