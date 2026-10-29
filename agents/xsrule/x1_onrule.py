"""(a) overnight rule signals: top-k LO / L/S, composites, market-relative. universe(N) from argv."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule")
from xlib import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
t0 = time.time(); D = load(N); print(f"{time.time()-t0:.0f}s", flush=True)
D = rlib.clean(D, [c for c in D.columns if c.startswith("z") or c in ("cc5","cc21","rv5_20","vday_rel")])
med = lambda c: D[c] - D.groupby("date")[c].transform("median")
base = 0.5 * D.z30 - D.z60
sigs = {
 "base": base,
 "base_mr": 0.5 * med("z30") - med("z60"),
 "z60only": -D.z60, "z30l": -D.z30l, "z120": -D.z120,
 "b_z15l": base - 0.5 * D.z15l, "b_z5l": base - 0.5 * D.z5l, "b_zmoc": base - 0.5 * D.zmoc,
 "b_zvwap": base - 0.5 * D.zvwap, "b_rng": base - 1.0 * (D.rngpos - 0.5), "b_cc5": base - 0.3 * D.cc5,
 "b_zon": base + 0.3 * D.zon_prev, "b_zoc": base - 0.3 * D.zoc,
 "b_vlast": base - 1.0 * D.r_vlast_share, "b_vmoc": base - 1.0 * D.r_vmoc_share,
 "b_zcc": base - 0.3 * D.zcc,
 "rank3": D.r_z30 * 0.5 - D.r_z60 - 0.5 * D.r_z15l,
 "rank4": D.r_z30 * 0.5 - D.r_z60 - 0.5 * D.r_z15l - 0.5 * D.r_zvwap,
 "rank5": D.r_z30 * 0.5 - D.r_z60 - 0.5 * D.r_z15l - 0.5 * D.r_zvwap - 0.5 * D.r_rngpos,
 "comp_a": base - 0.5 * D.z15l - 0.5 * D.zvwap,
 "comp_b": base - 0.5 * D.z15l - 0.5 * D.zvwap - 1.0 * (D.rngpos - 0.5),
 "comp_c": base - 0.5 * D.z15l - 0.5 * D.zvwap - 0.3 * D.cc5,
 "comp_d": base - 0.5 * D.z15l - 0.5 * D.zvwap - 0.3 * D.cc5 - 0.3 * D.zcc,
}
res = {}
for nm, s in sigs.items():
    for mode, k, smin in [("lo", 20, 1.0 if nm not in ("rank3","rank4","rank5") else 0.0), ("lo", 20, None), ("ls", 20, None)]:
        tag = f"{nm}_{mode}{k}" + ("_s1" if smin else "")
        r = rlib.book(D, s, k=k, mode=mode, smin=smin, name=tag, p=False); res[tag] = r
        st = stats(r, tag, p=False); oos, tab = fold_oos(r, p=False)
        print(f"{tag:26s} Sh {st['sharpe']:5.2f} OOS {sh(oos):5.2f} h {st['h1']:.2f}/{st['h2']:.2f} mdd {st['mdd']*100:5.1f}% minfold {tab.fold_sh.min():5.2f}", flush=True)
pd.DataFrame(res).to_parquet(f"{SCR}/x1_res_{N}.parquet")
print("family fold-oos (all LO s1):"); fold_oos({k: v for k, v in res.items() if k.endswith("lo20_s1")}, label="LO_s1 family")
fold_oos({k: v for k, v in res.items() if k.endswith("ls20")}, label="LS family")
fold_oos({k: v for k, v in res.items() if k.endswith("lo20")}, label="LO nosmin family")
