"""(a) confirmation of FIXED composites: LO/LS, 1.0 vs 1.5 bps, turnover, correlations, fold-OOS per candidate and family."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule")
from xlib import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
UNIS = [int(u) for u in sys.argv[2].split(",")] if len(sys.argv) > 2 else [150, N]
D = load(N); D = rlib.clean(D, ["z30", "z60", "z5l", "zmoc", "zon_prev"])
def comps(d):
    b = 0.5 * d.z30 - d.z60
    return {"base": b,
            "cA": b - 0.5 * d.z5l + 0.3 * d.zon_prev - 1.0 * d.r_vlast_share,                   # no closing-print term
            "cB": b - 0.3 * d.z5l - 0.3 * d.zmoc + 0.3 * d.zon_prev - 1.0 * d.r_vlast_share}   # with zmoc
res = {}
for uni in UNIS:
    d = D[D.tk.map(lambda t: rlib.RANK.get(t, 9999) <= uni)].copy(); print(f"=== universe {uni}: {d.tk.nunique()} names", flush=True)
    for nm, s in comps(d).items():
        for mode, k, smin in [("lo", 20, 1.0), ("lo", 20, 1.5), ("ls", 20, None)]:
            tag = f"u{uni}_{nm}_{mode}{k}" + (f"_s{smin}" if smin else "")
            r = rlib.book(d, s, k=k, mode=mode, smin=smin, p=False); r15 = rlib.book(d, s, k=k, mode=mode, smin=smin, cost=1.5, p=False)
            res[tag] = r; st = stats(r, tag, p=False); oos, tab = fold_oos(r, p=False)
            # turnover: overnight books are fully closed every morning -> traded notional = 2 x gross exposure per day
            gexp = (1.0 if mode == "lo" else 2.0)
            print(f"{tag:22s} Sh {st['sharpe']:5.2f} OOS {sh(oos):5.2f} @1.5bp {sh(r15):5.2f} h {st['h1']:.2f}/{st['h2']:.2f} mdd {st['mdd']*100:5.1f}% ret {st['ret']*100:5.1f}% vol {st['vol']*100:4.1f}% minfold {tab.fold_sh.min():5.2f} neg {(tab.fold_sh<0).sum()} | turn {2*gexp:.0f}x/day | {corr_table(r)}", flush=True)
pd.DataFrame(res).to_parquet(f"{SCR}/x4_res_{N}_{'_'.join(map(str,UNIS))}.parquet")
for uni in UNIS:
    fold_oos({k: v for k, v in res.items() if k.startswith(f"u{uni}_") and "_lo20" in k}, label=f"u{uni} LO family")
    fold_oos({k: v for k, v in res.items() if k.startswith(f"u{uni}_") and "_ls20" in k}, label=f"u{uni} LS family")
    fold_oos({k: v for k, v in res.items() if k.startswith(f"u{uni}_") and "_ls20" in k and "cA" in k}, label=f"u{uni} LS cA only")
