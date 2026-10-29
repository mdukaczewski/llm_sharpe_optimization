"""(a) capacity + (b) training window sweep: universe(1500), yearly retrain, f28cal, rank target, seed 0.
Each raw prediction saved as out/raw_{name}.npy aligned to load(1500) row order. Evaluated at k=10/20 eq at the commission."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml2")
from o2lib import *
NJ = int(os.environ.get("ONML2_NJOBS", "2"))
D = load(1500, extra=False)
for c in CAL: pass
# calendar features are built in olib.load only with extra=True; rebuild them cheaply here
g = D.groupby("date")
D["dow"] = D.date.dt.dayofweek.astype(np.float32)
D["dme"] = np.clip(((D.date + pd.offsets.BMonthEnd(0)) - D.date).dt.days, 0, 25).astype(np.float32)
D["qend"] = ((D.date.dt.month % 3 == 0) & (D.dme <= 2)).astype(np.float32)
big = (D.zgap.abs() > 2.5).astype(int); idx = np.arange(len(D))
last = pd.Series(np.where(big == 1, idx, np.nan)).groupby(D.tk).ffill(); first = pd.Series(idx).groupby(D.tk).transform("min")
D["dsgap"] = np.clip((idx - last.fillna(first).values), 0, 60).astype(np.float32)
D = rlib.clean(D, CAL)
log(f"sweep1: {D.tk.nunique()} names rows {len(D)} n_jobs {NJ}")
CFG = {
  "base":      dict(),
  "mcs500":    dict(params=dict(min_child_samples=500)),
  "mcs1000":   dict(params=dict(min_child_samples=1000)),
  "cs05":      dict(params=dict(colsample_bytree=0.5)),
  "roll3":     dict(window_years=3),
  "roll4":     dict(window_years=4),
  "hl2":       dict(halflife_years=2),
  "hl4":       dict(halflife_years=4),
  "L31_600":   dict(params=dict(num_leaves=31, n_estimators=600, learning_rate=0.02)),
  "L31_1000":  dict(params=dict(num_leaves=31, n_estimators=1000, learning_rate=0.02)),
  "L63_600":   dict(params=dict(num_leaves=63, n_estimators=600, learning_rate=0.02)),
  "L63_1000":  dict(params=dict(num_leaves=63, n_estimators=1000, learning_rate=0.02)),
}
only = sys.argv[1:] or list(CFG)
for name in only:
    kw = CFG[name]; fp = f"{OUT}/raw_{name}.npy"
    if os.path.exists(fp): raw = pd.Series(np.load(fp), index=D.index); log(f"{name}: loaded")
    else:
        t0 = time.time(); raw = wf2(D, F28CAL, "yr", "Y", seed=0, n_jobs=NJ, **kw); np.save(fp, raw.values.astype(np.float32))
        log(f"{name}: trained {time.time()-t0:.0f}s IC {ic(D, raw)}")
    p = xrank(D, raw)
    for k in [10, 20]:
        r, P = book3(D, p, k=k); ev(r, P, f"sweep1 {name} k{k}"); save(f"s1_{name}_k{k}", r, P)
    r, P = book3(D, p, k=20, weight="pred"); ev(r, P, f"sweep1 {name} k20p"); save(f"s1_{name}_k20p", r, P)
log("sweep1 done")
