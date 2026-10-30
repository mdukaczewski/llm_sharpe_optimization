#!/usr/bin/env python3
"""Step 5: regenerate and save the kept candidates (series NET of auction cost at c = 0.3 / spread fraction f and borrow,
BEFORE the $3 commission; the coordinator adds commission from the lo10 / ls10 suffix).  SPECS is filled from s1/s2 results."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3lib import *
log("s4_save start")
D = load_frame(600); sig0 = cax(D)
# name, universe, k, mode, smin, exit, f, liquidity term weight on rank(hs0931) (0 = none), note
SPECS = [
    ("xsr3_cAx_u150_moo_lo10", 150, 10, "lo", 1.5, "oo", 1.0, 0.0),
    ("xsr3_cAx_u150_moo_lo20", 150, 20, "lo", 1.5, "oo", 1.0, 0.0),
    ("xsr3_cAx_u300_moo_lo10", 300, 10, "lo", 1.5, "oo", 1.0, 0.0),
    ("xsr3_cAx_u150_moo_ls10", 150, 10, "ls", None, "oo", 1.0, 0.0),
    ("xsr3_cAx_u300_moo_ls10", 300, 10, "ls", None, "oo", 1.0, 0.0),
    ("xsr3_cAx_u150_x05_lo10", 150, 10, "lo", 1.5, "on", 0.5, 0.0),
]
SPECS = [s for s in SPECS if s[0] in set(sys.argv[1:])] if len(sys.argv) > 1 else SPECS
nf = f"{HERE}/cands/NOTES.txt"
if os.path.exists(nf) and len(sys.argv) == 1: os.remove(nf)
C = {}; npos = {}
for name, u, k, mode, smin, ex, f, wl in SPECS:
    sub = D["rank"] <= u; s = sig0.copy()
    if wl: s = s - wl * (D[sub].groupby("date").hs0931.rank(pct=True) - 0.5).reindex(D.index).fillna(0)
    P = positions(D[sub], s[sub], k=k, mode=mode, smin=smin); r = net(P, exit=ex, c=0.3, f=f); n = k * (2 if mode == "ls" else 1)
    exe = "MOC entry + official-open exit, auction cost c=0.3 both sides" if ex == "oo" else f"MOC entry (auction cost c=0.3) + 09:31-bar exit paying {f} x predicted NBBO half-spread"
    note = (f"overnight {'LO' if mode == 'lo' else 'L/S'} top{k}{' cAx>' + str(smin) if smin else ''} on {u} most liquid names; {exe}; "
            f"{'- ' + str(wl) + ' x rank(hs0931) in the composite; ' if wl else ''}borrow 1%; BEFORE commission ({comm_bps(n):.1f} bps/day at $3/$0.4M); all inputs known at 15:59")
    rlib.save_cand(LANE, name, r, note); C[name] = r - comm_bps(n) / 1e4; npos[name] = n
    d = rep(r, name, n); log(f"   corr {corrs(r)}  picks/day {len(P)/P.date.nunique():.1f}")
oos, tab = fold_oos(C, p=False); log(f"family fold-OOS over the saved set (at commission): {sh(oos):.2f} | picks {tab.pick.value_counts().to_dict()}")
lo = {k: v for k, v in C.items() if "lo" in k and "x05" not in k}; ls = {k: v for k, v in C.items() if "ls" in k}
if lo: oos, tab = fold_oos(lo, p=False); log(f"family fold-OOS LO moo books: {sh(oos):.2f} picks {tab.pick.value_counts().to_dict()}")
if ls: oos, tab = fold_oos(ls, p=False); log(f"family fold-OOS L/S moo books: {sh(oos):.2f} picks {tab.pick.value_counts().to_dict()}")
log("candidate-candidate correlations:\n" + pd.DataFrame(C).corr().round(2).to_string())
log("s4_save done")
