"""(a) round 2: combine only the helpful terms; k / smin variants; also on the 150-name universe for comparison."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule")
from xlib import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
D = load(N); D = rlib.clean(D, ["z30", "z60", "z5l", "z15l", "zmoc", "zon_prev", "zcc"])
def sig(d):
    b = 0.5 * d.z30 - d.z60
    return {"base": b,
     "c1_z5l_zon": b - 0.5 * d.z5l + 0.3 * d.zon_prev,
     "c2_zmoc_zon": b - 0.5 * d.zmoc + 0.3 * d.zon_prev,
     "c3_z5l_zon_vl": b - 0.5 * d.z5l + 0.3 * d.zon_prev - 1.0 * d.r_vlast_share,
     "c4_z5l_zmoc_zon": b - 0.3 * d.z5l - 0.3 * d.zmoc + 0.3 * d.zon_prev,
     "c5_z5l_zmoc_zon_vl": b - 0.3 * d.z5l - 0.3 * d.zmoc + 0.3 * d.zon_prev - 1.0 * d.r_vlast_share,
     "c6_z5l_vl": b - 0.5 * d.z5l - 1.0 * d.r_vlast_share,
     "c7_z5l_zon_w": b - 0.7 * d.z5l + 0.5 * d.zon_prev,
     "c8_nozmoc_all": b - 0.5 * d.z5l + 0.3 * d.zon_prev - 1.0 * d.r_vlast_share - 0.3 * d.z15l,
    }
res = {}
for uni in [N, 150]:
    d = D if uni == N else D[D.tk.map(lambda t: rlib.RANK.get(t, 9999) <= 150)].copy()
    print(f"=== universe {uni}: {d.tk.nunique()} names", flush=True)
    for nm, s in sig(d).items():
        for mode, k, smin in [("lo", 20, 1.0), ("lo", 20, 1.5), ("lo", 30, 1.0), ("lo", 10, 1.0), ("ls", 20, None), ("ls", 30, None)]:
            tag = f"u{uni}_{nm}_{mode}{k}" + (f"_s{smin}" if smin else "")
            r = rlib.book(d, s, k=k, mode=mode, smin=smin, p=False); res[tag] = r
            st = stats(r, tag, p=False); oos, tab = fold_oos(r, p=False)
            print(f"{tag:34s} Sh {st['sharpe']:5.2f} OOS {sh(oos):5.2f} h {st['h1']:.2f}/{st['h2']:.2f} mdd {st['mdd']*100:5.1f}% minfold {tab.fold_sh.min():5.2f} | c {corr_table(r)}", flush=True)
pd.DataFrame(res).to_parquet(f"{SCR}/x1b_res_{N}.parquet")
for uni in [N, 150]:
    fold_oos({k: v for k, v in res.items() if k.startswith(f"u{uni}_") and "_lo20_s1.0" in k}, label=f"u{uni} LO20 s1 family")
    fold_oos({k: v for k, v in res.items() if k.startswith(f"u{uni}_") and "_ls20" in k}, label=f"u{uni} LS20 family")
    fold_oos({k: v for k, v in res.items() if k.startswith(f"u{uni}_") and "_lo" in k}, label=f"u{uni} all LO family")
