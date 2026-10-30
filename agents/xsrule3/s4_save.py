#!/usr/bin/env python3
"""Step 5: regenerate and save the kept candidates (series NET of auction cost at c = 0.3 / spread fraction f and borrow,
BEFORE the $3 commission; the coordinator adds commission from the lo10 / ls10 suffix).  SPECS is filled from s1/s2 results."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3lib import *
log("s4_save start")
D = load_frame(600); sig0 = cax(D)
# name, universe, k, mode, smin, exit, f, cap on predicted hs0931 (bps; 0 = no filter)
SPECS = [
    ("xsr3_cAx_u150_moo_lo10", 150, 10, "lo", 1.5, "oo", 1.0, 0),
    ("xsr3_cAx_u150_moo_lo20", 150, 20, "lo", 1.5, "oo", 1.0, 0),
    ("xsr3_cAx_u50_moo_lo10", 50, 10, "lo", 1.5, "oo", 1.0, 0),
    ("xsr3_cAx_u150_fhs10_moo_lo10", 150, 10, "lo", 1.5, "oo", 1.0, 10),
    ("xsr3_cAx_u150_fhs10_moo_ls10", 150, 10, "ls", None, "oo", 1.0, 10),
    ("xsr3_cAx_u150_x05_lo10", 150, 10, "lo", 1.5, "on", 0.5, 0),
]
SPECS = [s for s in SPECS if s[0] in set(sys.argv[1:])] if len(sys.argv) > 1 else SPECS
nf = f"{HERE}/cands/NOTES.txt"
if os.path.exists(nf) and len(sys.argv) == 1: os.remove(nf)
C = {}; npos = {}
for name, u, k, mode, smin, ex, f, wl in SPECS:
    sub = D["rank"] <= u; s = sig0
    P = positions(D[sub], s[sub], k=k, mode=mode, smin=smin, filt=None if not wl else (D.hs0931 <= wl)[sub]); r = net(P, exit=ex, c=0.3, f=f); n = k * (2 if mode == "ls" else 1)
    exe = "MOC entry + official-open exit, auction cost c=0.3 both sides" if ex == "oo" else f"MOC entry (auction cost c=0.3) + 09:31-bar exit paying {f} x predicted NBBO half-spread"
    note = (f"overnight {'LO' if mode == 'lo' else 'L/S'} top{k}{' cAx>' + str(smin) if smin else ''} on {u} most liquid names; {exe}; "
            f"{'names with predicted 09:31 half-spread > ' + str(wl) + ' bps skipped before ranking; ' if wl else ''}borrow 1%; BEFORE commission ({comm_bps(n):.1f} bps/day at $3/$0.4M); all inputs known at 15:59")
    rlib.save_cand(LANE, name, r, note); C[name] = r - comm_bps(n) / 1e4; npos[name] = n
    d = rep(r, name, n); log(f"   corr {corrs(r)}  picks/day {len(P)/P.date.nunique():.1f}  hs0931 mean {P.hs0931.mean():.1f}  cost/side c=0.3 in {cost_sides(P,0.3)[0].mean():.2f} out {cost_sides(P,0.3)[1].mean():.2f}")
    oos, tab = fold_oos(C[name], p=False); log("   folds: " + " ".join(f"{x:+.1f}" for x in tab.fold_sh))
    log("   tiers:\n" + tier_table(P, 0.3, "oo" if ex == "oo" else "on").round(2).to_string(index=False))
    sens = {lab: rep(net(P, **kw), f"{name} [{lab}]", n, p=False) for lab, kw in [("c=0.1", dict(exit="oo", c=0.1)), ("c=0.3", dict(exit="oo", c=0.3)), ("c=1.0", dict(exit="oo", c=1.0)), ("x1", dict(exit="on", f=1.0)), ("x05", dict(exit="on", f=0.5)), ("old 1bp", dict(old=True))]}
    log("   sensitivity (Sh / OOS at commission): " + "  ".join(f"{lab} {v['sh']:.2f}/{v['oos']:.2f}" for lab, v in sens.items()))
oos, tab = fold_oos(C, p=False); log(f"family fold-OOS over the saved set (at commission): {sh(oos):.2f} | picks {tab.pick.value_counts().to_dict()}")
lo = {k: v for k, v in C.items() if "lo" in k and "x05" not in k}; ls = {k: v for k, v in C.items() if "ls" in k}
if lo: oos, tab = fold_oos(lo, p=False); log(f"family fold-OOS LO moo books: {sh(oos):.2f} picks {tab.pick.value_counts().to_dict()}")
if ls: oos, tab = fold_oos(ls, p=False); log(f"family fold-OOS L/S moo books: {sh(oos):.2f} picks {tab.pick.value_counts().to_dict()}")
log("candidate-candidate correlations:\n" + pd.DataFrame(C).corr().round(2).to_string())
log("s4_save done")
