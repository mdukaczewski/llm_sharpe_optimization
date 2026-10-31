"""Regenerate the 4 dml2 candidates with STRAT_END=2026-10-28 (same code path as s1_train.py b5 / b10 + s5_final.py).
No metrics are computed or printed."""
import os, sys, time
assert os.environ.get("STRAT_END") == "2026-10-28"
sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dml2")
from dlib import *
from s2_port import score
from s5_final import CANDS
def say(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True)
t0 = time.time()
P, F, m = setup()
# dev-identical continuation: tickers that enter load_sub's keep set only because the panel now runs to 2026-10-28
# (keep = ever adv20>=50M & ADJUSTED close>=5 up to the panel end) are masked out on dates <= 2026-04-27
from bt import rolling_mean
_z = np.load(f"{rlib.DATA}/panel_liquid.npz", allow_pickle=True); _d = pd.to_datetime(_z["dates"]); _T0 = int(np.searchsorted(_d, pd.Timestamp("2026-04-27"), side="right"))
_adj = _z["adj"][:_T0].astype(np.float64); _c = _z["close"][:_T0].astype(np.float64) * _adj; _v = _z["vol"][:_T0].astype(np.float64) / _adj
_k = (np.isin(_z["ttype"], ["CS", "ADRC"])[None, :] & (rolling_mean(_c * _v, 20, 15) >= 50e6) & (_c >= 5.0)).any(axis=0); _k[list(_z["tickers"]).index("SPY")] = True
_dev = set(_z["tickers"][_k]); _new = np.array([t not in _dev for t in P.tickers]); del _adj, _c, _v, _z
say(f"dev keep set {len(_dev)} tickers; {int(_new.sum())} new tickers; dev mask cells removed {int(m[:_T0][:, _new].sum())}")
m[:_T0, _new] = False
rt, ri = np.nonzero(m); rows = (rt, ri)
say(f"setup done T={P.T} N={P.N} last {P.dates[-1].date()} {time.time()-t0:.0f}s")
names, X = build_features(P, F, m, rows)
say(f"features {X.shape} {time.time()-t0:.0f}s")
y = fwd_target(P, F, m, 10, 0); pred, _ = walk_forward(P, m, rows, X, names, y, 10, retrain=126); save_pred(pred, "b10_r126_f32"); say("b10 f32 saved")
del X; import gc; gc.collect()
