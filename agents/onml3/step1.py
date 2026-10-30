#!/usr/bin/env python3
"""Step 1: the saved f26calx 3-seed quarterly predictions (no refit) under every execution case.
(a) y_on, rlib costs (1-2.5 bps/side)        (b) y_oo, MOC+MOO auction cost at c = 0.1/0.3/1.0 + borrow
(c) y_on, MOC entry (auction c=0.3) + crossing the 09:31 half-spread on the exit (hs0931)
(d) same, paying half the half-spread.  All at k=10 (plus k sweep for (b) at c=0.3), $0.4M commission."""
from lib3 import *
t0 = time.time(); D = frame(2000); P = load_pred()
G = D.merge(P, on=["date", "tk"], how="inner"); log(f"step1: merged {len(G)} rows, {G.date.nunique()} days, {time.time()-t0:.0f}s")
G["c0"] = G.tk.map(rlib.cost_bps).astype(np.float32); G["zero"] = np.float32(0.0)
G["hsx1"] = G.hs0931; G["hsx05"] = 0.5 * G.hs0931
rows = []; keep = {}
def run(name, k=10, **kw):
    r, L = book3(G, k=k, ret_pos=True, **kw); m = metrics(r, k); log("  " + mline(name, m)); rows.append((name, k) + tuple(round(float(m[x]), 3) for x in ["sh_pre", "sh", "fold", "h1", "h2", "mdd", "ret"]))
    keep[name] = (r, L); return r, L
log("--- (a) old target y_on, rlib costs")
r, L = run("a_yon_rlib_k10", y="y_on", ce="c0", cx="c0")
old = old_cand(); log(f"  check vs saved cand onml_f26calx_u2000_q3_k10: corr {corr_with(r, old):.4f} mean diff {(r-old).dropna().abs().mean()*1e4:.3f} bps")
log("--- (a') y_on gross only (no cost) and y_oo gross only")
run("a_yon_gross_k10", y="y_on", ce="zero", cx="zero", borrow=0.0); run("b_yoo_gross_k10", y="y_oo", ce="zero", cx="zero", borrow=0.0)
log("--- (b) y_oo, MOC entry + MOO exit, auction cost c = 0.1 / 0.3 / 1.0 + borrow 1%")
for c in [0.1, 0.3, 1.0]:
    cost_cols(G, k=10, c=c); run(f"b_yoo_moo_c{c}_k10", y="y_oo", ce="ce", cx="cx")
log("--- (b) k sweep at c=0.3")
for k in [6, 8, 12, 15, 20]:
    cost_cols(G, k=k, c=0.3); run(f"b_yoo_moo_c0.3_k{k}", k=k, y="y_oo", ce="ce", cx="cx")
cost_cols(G, k=10, c=0.3)
log("--- (c)/(d) y_on, MOC entry (auction c=0.3) + 09:31 exit paying f x hs0931, f = 1 / 0.5 / 0.25; also f=1 with no entry cost")
run("c_yon_x1_k10", y="y_on", ce="ce", cx="hsx1"); run("d_yon_x05_k10", y="y_on", ce="ce", cx="hsx05")
G["hsx025"] = 0.25 * G.hs0931; run("d_yon_x025_k10", y="y_on", ce="ce", cx="hsx025"); run("c_yon_x1_noentry_k10", y="y_on", ce="zero", cx="hsx1")
log("--- by tier (positions / bps per trade / PnL share), k=10")
for name in ["a_yon_rlib_k10", "b_yoo_moo_c0.3_k10", "c_yon_x1_k10", "d_yon_x05_k10"]:
    r, L = keep[name]; y = "y_on" if "yon" in name else "y_oo"; log(f"  {name}\n" + tier_table(L, y).round(2).to_string(index=False))
log("--- cost summary of the k=10 book's positions (c=0.3): mean ce / cx bps per side, hs0931")
L = keep["b_yoo_moo_c0.3_k10"][1]; log(f"  ce mean {L.ce.mean():.2f} median {L.ce.median():.2f} | cx mean {L.cx.mean():.2f} median {L.cx.median():.2f} p90 {L.cx.quantile(.9):.2f}")
L2 = keep["c_yon_x1_k10"][1]; log(f"  hs0931 of chosen names: mean {L2.hsx1.mean():.2f} median {L2.hsx1.median():.2f}")
T = pd.DataFrame(rows, columns=["book", "k", "sh_pre", "sh_04M", "fold_04M", "h1", "h2", "mdd", "ret"]); T.to_csv(f"{OUT}/step1.csv", index=False)
log("STEP1 DONE\n" + T.to_string(index=False))
