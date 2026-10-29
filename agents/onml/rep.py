"""Report table for saved candidates (agents/onml/cands) + chosen out/ series."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
names = sys.argv[1:]
C = rlib.load_cands("onml") if not names else {n: pd.read_csv(f"{OUT}/{n}.csv", index_col=0, parse_dates=True).iloc[:, 0] for n in names}
sl = sleeve(); sp = spy_cc()
print(f"{'name':36s} {'full':>5s} {'fOOS':>5s} {'h1':>5s} {'h2':>5s} {'mdd':>6s} {'ret':>6s} {'vol':>5s} {'c_slv':>5s} {'c_spy':>5s} {'min_fold':>8s}")
for n, r in C.items():
    s = rlib.stats(r, p=False); o, t = rlib.fold_oos({"x": r}, p=False)
    j = pd.concat([r, sl, sp], axis=1).dropna()
    print(f"{n:36s} {s['sharpe']:5.2f} {sh(o):5.2f} {s['h1']:5.2f} {s['h2']:5.2f} {s['mdd']*100:5.1f}% {s['ret']*100:5.1f}% {s['vol']*100:4.1f}% {j.iloc[:,0].corr(j.iloc[:,1]):5.2f} {j.iloc[:,0].corr(j.iloc[:,2]):+5.2f} {t.fold_sh.min():8.2f}")
rlib.fold_oos(C, label="candidate family")
rlib.fold_oos({"sleeve": sl}, label="existing sleeve onlgb1200q_ls20")
