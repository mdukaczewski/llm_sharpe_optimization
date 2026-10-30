#!/usr/bin/env python3
"""Step 5: final configurations (quarterly 3-seed predictions) -> candidates through the UNCHANGED onml4 pipeline.
Series = net of the route's modelled cost at c = 0.3 / f = 1 and borrow, BEFORE the base commission; decision-dated (rlib.book
convention; folds.py re-dates the onml family).  Plus c / f sensitivity, old convention, tiers, correlations, family fold-OOS.
usage: python3 step5.py [save=1]      configs in final_cfg.txt: name|pred spec|k|lam|entry|exit|note"""
from lib5 import *
# measured adverse selection of the 15:59:30 cross (agents/onml4x): +1.85 bps per cross entry, in selection and charge
ADV = float(os.environ.get("ADV", "1.85")); _rc = lib4.route_costs
def _route_costs(G, k, c=0.3, f=1.0, ls=True):
    C = _rc(G, k, c, f, ls); C["x1559"] = (C["x1559"][0] + ADV, C["x1559"][1] + ADV); return C
lib4.route_costs = _route_costs
kw = dict(a.split("=") for a in sys.argv[1:] if "=" in a); SAVE = kw.get("save") == "1"
G = load_G(); G["pR"] = prank((G.p40 + G.p26) / 2, G.date)
def loadp(name):
    P = pd.read_parquet(f"{OUT}/pred_{name}.parquet", columns=["date", "tk", "p"]); s = G[["date", "tk"]].merge(P, on=["date", "tk"], how="left").p
    s.index = G.index; return prank(s, G.date)
G["p40n"] = loadp("q_f40n" if os.path.exists(f"{OUT}/pred_q_f40n.parquet") else "q_f40n_s0")   # 3-seed run was cut after seed 0
if os.path.exists(f"{OUT}/pred_q_f26n.parquet"): G["p26n"] = loadp("q_f26n"); G["pRnn"] = prank((G.p40n + G.p26n) / 2, G.date)
G["pRn"] = prank((G.p40n + G.p26) / 2, G.date)              # new f40n replaces p40 in the onml4 rank average
G["pR3"] = prank((G.p40 + G.p26 + G.p40n) / 3, G.date)       # does it add to the existing pR?
CFG = [l.split("|", 6) for l in open(f"{LANE}/final_cfg.txt").read().strip().splitlines() if l.strip() and not l.startswith("#")]
REF = load_ref(REF_RTX); REFM = load_ref(REF_MOO)
# IC table of the final prediction sets
ics = []
for p in ["p40", "p26", "pR", "p40n", "pRn", "pR3"] + (["p26n", "pRnn"] if "p26n" in G else []):
    row = dict(pred=p); v = daily_ic(G, p, "y_oo"); row["all"] = round(v.mean(), 4)
    v = daily_ic(G, p, "y_oo", by="tier").reset_index(); v.columns = ["date", "tier", "ic"]
    for t, x in v.groupby("tier"): row[TNAMES[t]] = round(x.ic.mean(), 4)
    x = v[v.tier == 0]; x = x.assign(y=x.date.dt.year); row["1-300 by yr"] = " ".join(f"{a:.3f}" for a in x.groupby("y").ic.mean())
    ics.append(row)
tlog("STEP5 IC vs y_oo (per-day Spearman, within tier)", pd.DataFrame(ics))
Hs = {}; rows = []; sens = []; tiers = []; series = {}
for name, pc, k, lam, ent, ex, note in CFG:
    k = int(k); lam = float(lam)
    if pc not in G: log(f'skip {name}: {pc} not available'); continue
    if pc not in Hs: Hs[pc] = prep(G, pc)
    H = Hs[pc]
    r, L = route_book(H, k, lam, ent, ex, 0.3, 1.0, ret_pos=True); series[name] = r
    m = metrics(r, k); row = mrow(name, m); rn = commission(r, k)
    row.update(corr_rtx=round(corr_with(rn, commission(REF, 6)), 2), corr_moo=round(corr_with(rn, commission(REFM, 6)), 2),
               gross_bps=round((np.sign(L.w) * L.y).mean() * 1e4, 2), cost_bps=round(L.c.mean(), 2),
               entry_x1559=round((L.entry == "x1559").mean(), 2), exit_moo=round((L.exit == "moo").mean(), 2), t1_pct=round((L.tier == 0).mean() * 100, 1))
    rows.append(row); log(str({k_: v for k_, v in row.items() if k_ != "yrs"}))
    t = tier_table(L); t.insert(0, "book", name); tiers.append(t)
    for c, f in [(0.1, 1.0), (1.0, 1.0), (0.3, 0.5), (0.3, 2.0)]:
        ra = route_book(H, k, lam, ent, ex, c, f, c_sel=0.3, f_sel=1.0); rb = route_book(H, k, lam, ent, ex, c, f)
        ma, mb = metrics(ra, k), metrics(rb, k)
        sens.append(dict(book=name, c=c, f=f, fixed_sh=round(ma["sh"], 2), fixed_fold=round(ma["fold"], 2), resel_sh=round(mb["sh"], 2), resel_fold=round(mb["fold"], 2)))
    Lo = L[["date", "tk", "w"]].merge(H[["date", "tk", "y_on"]], on=["date", "tk"], how="left")
    Lo["c"] = Lo.tk.map(rlib.cost_bps); Lo["n"] = Lo.w * Lo.y_on.fillna(0) - Lo.w.abs() * 2 * Lo.c / 1e4 - Lo.w.clip(upper=0).abs() * 0.01 / 252
    mo = metrics(Lo.groupby("date").n.sum(), k); sens.append(dict(book=name, c="old", f="1-2.5bp 09:31", fixed_sh=round(mo["sh"], 2), fixed_fold=round(mo["fold"], 2), resel_sh=np.nan, resel_fold=np.nan))
    if SAVE and name.startswith("onml5_"): rlib.save_cand("onml5", name, r, note)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/step5_final.csv", index=False)
tlog("STEP5 final (c=0.3, f=1; $0.4M commission)", T.drop(columns=["yrs"]))
tlog("STEP5 by year ($0.4M)", T[["book", "yrs"]])
tlog("STEP5 sensitivity (fixed = selection at c=0.3/f=1 then charged; resel = selection re-run with that cost)", pd.DataFrame(sens))
tlog("STEP5 tiers (c=0.3)", pd.concat(tiers))
C = pd.DataFrame({k_: commission(v, int(k_.split("_k")[-1])) for k_, v in series.items()}); C["onml4_pR_rtx_k6"] = commission(REF, 6); C["onml4_pR_moo_k6"] = commission(REFM, 6)
log("--- STEP5 correlation matrix ($0.4M)\n" + C.corr().round(2).to_string())
fam = {k_: commission(v, int(k_.split("_k")[-1])) for k_, v in series.items()}
sv = [n for n in fam if n.startswith("onml5_")]
fo, tab = rlib.fold_oos({n: fam[n] for n in sv}, p=False); log(f"STEP5 family fold-OOS over {len(sv)} candidates: {sh(fo):.2f}; picks {tab.pick.value_counts().to_dict()}")
o4 = {os.path.basename(p)[:-4]: load_ref(p.replace(HERE + "/", "")) for p in sorted(__import__("glob").glob(f"{HERE}/agents/onml4x/cands/*.csv"))}
o4 = {n: commission(v, int(n.split("_k")[-1])) for n, v in o4.items()}
fo, tab = rlib.fold_oos(o4, p=False); log(f"STEP5 onml4 family fold-OOS (reference): {sh(fo):.2f}")
fam2 = dict(o4); fam2.update({n: fam[n] for n in sv}); fo, tab = rlib.fold_oos(fam2, p=False)
log(f"STEP5 onml4 + onml5 family fold-OOS: {sh(fo):.2f}; picks {tab.pick.value_counts().to_dict()}")
log("STEP5 DONE")
