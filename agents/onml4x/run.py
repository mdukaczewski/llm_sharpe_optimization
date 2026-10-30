#!/usr/bin/env python3
"""onml4x: the onml4 pR books re-run with the MEASURED adverse selection of the 15:59:30 cross added to its cost.
agents/onml4/cross_check_positions.py: on the quote-panel days the real 15:59:30 ask/bid of the book's cross entries is 1.22 bps
(t 3.7) worse than the modelled c389 x (1 +- hs1559) fill (longs 1.85, shorts 0.60).  Here the cross costs hs1559 + 1.85 bps
for longs-and-shorts alike (conservative: the long-side figure) in both selection and charge; MOC is unchanged.
Series conventions as onml4 (net of route cost at c = 0.3 / f = 1 and borrow, before commission, decision-dated)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml4")
import lib4
from lib4 import *
ADV = float(sys.argv[1]) if len(sys.argv) > 1 else 1.85
_rc = lib4.route_costs
def route_costs(G, k, c=0.3, f=1.0, ls=True):
    C = _rc(G, k, c, f, ls); C["x1559"] = (C["x1559"][0] + ADV, C["x1559"][1] + ADV); return C
lib4.route_costs = route_costs
G = load_G(); G["pR"] = (((G.p40 + G.p26) / 2).groupby(G.date).rank(pct=True) - 0.5).astype(np.float32)
H = prep(G, "pR"); sh = rlib.sh
for name, ent, ex in [("onml4x_pR_lam3_rte_moo_k6", "route", "moo"), ("onml4x_pR_lam3_rte_rtx_k6", "route", "route"), ("onml4x_pR_lam3_moc_moo_k6", "moc", "moo")]:
    r, L = route_book(H, 6, 3.0, ent, ex, 0.3, 1.0, ret_pos=True); rn = commission(r, 6)
    xs = (L.entry == "x1559").mean() * 100
    print(f"{name:30s} $0.4M Sh {sh(rn):.2f} fold-OOS {sh(rlib.fold_oos(rn, p=False)[0]):.2f} | crosses {xs:.0f}% of entries", flush=True)
    if len(sys.argv) <= 1:
        rlib.save_cand("onml4x", name, r, f"onml4 {name.replace('onml4x_','')} config with the measured 15:59:30-cross adverse selection (+{ADV} bps per cross entry, NBBO panel) in selection and charge; c=0.3, f=1; net of route cost + borrow, before commission")
