#!/usr/bin/env python3
"""Step 4: final configurations -> candidates (series net of their own route's modelled cost at c = 0.3 / f = 1, borrow and
the extra commission of split legs; BEFORE the base commission).  Plus c / f sensitivity, old convention, tiers, correlations.
usage: python3 step4.py save=1"""
from lib4 import *
kw = dict(a.split("=") for a in sys.argv[1:] if "=" in a); SAVE = kw.get("save") == "1"
G = load_G()
def prank(s, d): return (s.groupby(d).rank(pct=True) - 0.5).astype(np.float32)
G["pR"] = prank((G.p40 + G.p26) / 2, G.date)
# (name, pred, k, lam, entry, exit, note)
CFG = [l.split("|", 6) for l in open(f"{LANE}/final_cfg.txt").read().strip().splitlines() if l.strip() and not l.startswith("#")]
REFA = load_ref("agents/onml3/cands/onml3_f26calx_q3_lam3_moo_k6.csv"); REFB = load_ref("agents/onml3/cands/onml3_f40_on_lam3_moo_k6.csv")
REFX = load_ref("agents/xsrule3/cands/xsr3_cAx_u50_moo_lo10.csv")
Hs = {}; rows = []; sens = []; tiers = []; series = {}
for name, pc, k, lam, ent, ex, note in CFG:
    k = int(k); lam = float(lam)
    if pc not in Hs: Hs[pc] = prep(G, pc)
    H = Hs[pc]
    r, L = route_book(H, k, lam, ent, ex, 0.3, 1.0, ret_pos=True); series[name] = r
    m = metrics(r, k); row = mrow(name, m); rn = commission(r, k)
    row.update(corr_A=round(corr_with(rn, commission(REFA, 6)), 2), corr_B=round(corr_with(rn, commission(REFB, 6)), 2), corr_xsr=round(corr_with(rn, REFX - 10 * 2 * 3 / CAP), 2),
               gross_bps=round((np.sign(L.w) * L.y).mean() * 1e4, 2), cost_bps=round(L.c.mean(), 2), xo=round(L.xo.mean(), 3),
               entry_x1559=round((L.entry == "x1559").mean(), 2), exit_moo=round((L.exit == "moo").mean(), 2))
    rows.append(row); log(str({k_: v for k_, v in row.items() if k_ != "yrs"}))
    t = tier_table(L); t.insert(0, "book", name); tiers.append(t)
    # sensitivity: (a) selection fixed at c=0.3/f=1, charged at c / f; (b) selection re-run at that c / f
    for c, f in [(0.1, 1.0), (1.0, 1.0), (0.3, 0.5), (0.3, 2.0)]:
        ra = route_book(H, k, lam, ent, ex, c, f, c_sel=0.3, f_sel=1.0); rb = route_book(H, k, lam, ent, ex, c, f)
        ma, mb = metrics(ra, k), metrics(rb, k)
        sens.append(dict(book=name, c=c, f=f, fixed_sh=round(ma["sh"], 2), fixed_fold=round(ma["fold"], 2), resel_sh=round(mb["sh"], 2), resel_fold=round(mb["fold"], 2)))
    # old convention: same positions, 09:31-bar exit, 1 / 1.5 / 2.5 bps per side
    Lo = L[["date", "tk", "w"]].merge(H[["date", "tk", "y_on"]], on=["date", "tk"], how="left")
    Lo["c"] = Lo.tk.map(rlib.cost_bps); Lo["n"] = Lo.w * Lo.y_on.fillna(0) - Lo.w.abs() * 2 * Lo.c / 1e4 - Lo.w.clip(upper=0).abs() * 0.01 / 252
    ro = Lo.groupby("date").n.sum(); mo = metrics(ro, k); sens.append(dict(book=name, c="old", f="1-2.5bp 09:31", fixed_sh=round(mo["sh"], 2), fixed_fold=round(mo["fold"], 2), resel_sh=np.nan, resel_fold=np.nan))
    if SAVE: rlib.save_cand("onml4", name, r, note)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/step4_final.csv", index=False)
tlog("STEP4 final candidates (c=0.3, f=1; $0.4M commission)", T.drop(columns=["yrs"]))
tlog("STEP4 by year ($0.4M)", T[["book", "yrs"]])
tlog("STEP4 sensitivity (fixed = selection at c=0.3/f=1 then charged; resel = selection re-run with that cost)", pd.DataFrame(sens))
tlog("STEP4 tiers (c=0.3)", pd.concat(tiers))
C = pd.DataFrame({k_: commission(v, int(k_.split("_k")[-1])) for k_, v in series.items()}); C["onml3_A_k6"] = commission(REFA, 6); C["onml3_B_k6"] = commission(REFB, 6); C["xsr3_u50"] = REFX - 10 * 2 * 3 / CAP
log("--- STEP4 correlation matrix ($0.4M)\n" + C.corr().round(2).to_string())
fam = {k_: commission(v, int(k_.split("_k")[-1])) for k_, v in series.items()}
fo, tab = rlib.fold_oos(fam, p=False); log(f"STEP4 family fold-OOS over the {len(fam)} candidates: {sh(fo):.2f}; picks {tab.pick.value_counts().to_dict()}")
fam2 = dict(fam); fam2.update({"onml3_A": commission(REFA, 6), "onml3_B": commission(REFB, 6)})
fo, tab = rlib.fold_oos(fam2, p=False); log(f"STEP4 family fold-OOS with the two onml3 k6 books added: {sh(fo):.2f}; picks {tab.pick.value_counts().to_dict()}")
log("STEP4 DONE")
