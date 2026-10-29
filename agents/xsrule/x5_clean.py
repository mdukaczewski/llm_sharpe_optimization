"""(a) clean (pre-close) composite: last-hour volume share EX closing auction; compare with the auction-inclusive one."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule")
from xlib import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
UNIS = [int(u) for u in sys.argv[2].split(",")] if len(sys.argv) > 2 else [150, N]
D = load(N); D = add_extra2(D); D = rlib.clean(D, ["z30", "z60", "z5l", "zmoc", "zon_prev"])
print("corr vlast_share vs vlast_x:", round(D.vlast_share.corr(D.vlast_x), 3), flush=True)
def comps(d):
    b = 0.5 * d.z30 - d.z60
    return {"cA": b - 0.5 * d.z5l + 0.3 * d.zon_prev - 1.0 * d.r_vlast_share,
            "cAx": b - 0.5 * d.z5l + 0.3 * d.zon_prev - 1.0 * d.r_vlast_x,
            "cAx15": b - 0.5 * d.z5l + 0.3 * d.zon_prev - 1.0 * d.r_v15_x,
            "cAx_novl": b - 0.5 * d.z5l + 0.3 * d.zon_prev,
            "cBx": b - 0.3 * d.z5l - 0.3 * d.zmoc + 0.3 * d.zon_prev - 1.0 * d.r_vlast_x}
res = {}
for uni in UNIS:
    d = D[D.tk.map(lambda t: rlib.RANK.get(t, 9999) <= uni)].copy(); print(f"=== universe {uni}: {d.tk.nunique()} names", flush=True)
    for nm, s in comps(d).items():
        for mode, k, smin in [("lo", 20, 1.5), ("ls", 20, None)]:
            tag = f"u{uni}_{nm}_{mode}{k}" + (f"_s{smin}" if smin else "")
            r = rlib.book(d, s, k=k, mode=mode, smin=smin, p=False); r15 = rlib.book(d, s, k=k, mode=mode, smin=smin, cost=1.5, p=False)
            res[tag] = r; st = stats(r, tag, p=False); oos, tab = fold_oos(r, p=False)
            print(f"{tag:22s} Sh {st['sharpe']:5.2f} OOS {sh(oos):5.2f} @1.5bp {sh(r15):5.2f} h {st['h1']:.2f}/{st['h2']:.2f} mdd {st['mdd']*100:5.1f}% minfold {tab.fold_sh.min():5.2f} neg {(tab.fold_sh<0).sum()} | {corr_table(r)}", flush=True)
pd.DataFrame(res).to_parquet(f"{SCR}/x5_res_{N}.parquet")
