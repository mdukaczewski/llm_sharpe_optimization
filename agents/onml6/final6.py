#!/usr/bin/env python3
"""onml6 final books.  usage: python3 final6.py <Y|Q> [save=1]
Books (k6, lam 3, MOO exit, c = 0.3 / f = 1 central, onml4x cross adverse selection +1.85 bps):
  ref     pR_o route  : the onml4x book (original predictions: features to the 15:59 bar AND the official close) - infeasible
  ref     pR_388 route: features to 15:59:00 for both legs - still infeasible for the MOC leg (diagnostic of the official-close part)
  (a)     pR_378 moc  : MOC only, decided at 15:49:00
  (b)     two         : MOC at 15:49 on pR_378 for MOC-class names, the rest 15:59:30 crosses on pR_388 (book6.two_stage)
  (b3)    two m3      : fixed mix, 3 MOC names per side at 15:49 + 3 crosses per side at 15:59 (book6.two_stage mocslots=3)
  x1559   pR_388 x1559: every entry a 15:59:30 cross on 15:59:00 information (feasible)
  lmt     LOC at the estimated 15:49 bid / ask on the (a) positions (lmt6)"""
from lmt6 import *
import glob
V = sys.argv[1]; SAVE = "save=1" in sys.argv; K = 6; LAM = 3.0
O3 = f"{HERE}/agents/onml3/out"
if V == "Y":
    preds = {"o40": [f"{O3}/pred_y_f40_on.parquet"], "o26": [f"{O3}/pred_y_f26x_on.parquet"]}
    for c in (378, 388):
        for s in ("f40", "f26"): preds[f"p{s[1:]}_{c}"] = [f"{OUT}/pred_{s}_c{c}_Y_s0.parquet"]
else:
    preds = {"o40": [f"{O3}/pred_f40_on.parquet"], "o26": [f"{O3}/pred_f26calx_q3.parquet"], "o40s0": [f"{O3}/pred_f40_on_q_s0.parquet"]}
    for c in (378, 388):
        for s in ("f40", "f26"):
            preds[f"p{s[1:]}_{c}"] = sorted(glob.glob(f"{OUT}/pred_{s}_c{c}_Q_s*.parquet"))
        preds[f"p40s0_{c}"] = [f"{OUT}/pred_f40_c{c}_Q_s0.parquet"]
H0 = load_H({}); H0 = H0.drop(columns=["cAx", "zcAx", "y_0935", "y_1031"])
for kcol, fs in preds.items():
    for i, f in enumerate(fs):
        P = pd.read_parquet(f, columns=["date", "tk", "p"]).rename(columns={"p": f"_s{i}"}); H0 = H0.merge(P, on=["date", "tk"], how="left")
    cols = [f"_s{i}" for i in range(len(fs))]
    H0[kcol] = H0[cols[0]] if len(cols) == 1 else prank(H0[cols].mean(1, skipna=False), H0.date); H0 = H0.drop(columns=cols)
    log(f"final6 {V}: {kcol} <- {len(fs)} file(s) {[os.path.basename(f) for f in fs]}")
H0["hs1559_orig"] = H0.hs1559
for c_ in H0.columns:
    if H0[c_].dtype == np.float64: H0[c_] = H0[c_].astype(np.float32)
def pr(a, b): return prank((H0[a] + H0[b]) / 2, H0.date)
H0["pR_o"] = pr("o40", "o26"); H0["pR_378"] = pr("p40_378", "p26_378"); H0["pR_388"] = pr("p40_388", "p26_388")
pcols = [c for c in H0.columns if c.startswith(("o40", "o26", "p40", "p26", "pR_"))]
H0 = H0[H0[pcols].notna().all(1)].reset_index(drop=True)
log(f"final6 {V}: {len(H0)} rows {H0.date.min().date()}..{H0.date.max().date()}")
for p in pcols: log(f"  IC y_oo {p}: {rlib.ic(H0, H0[p], 'y_oo')[0]:.4f}")
REF = ref_series(); REFn = commission(REF, K)
rows = []; S = {}; POS = {}
def rec(name, r, L=None, extra=None, comm=True):
    rn = commission(r, K) if comm else r
    m = full_metrics(r, K) if comm else lmetrics(r)
    row = mrow(name, m) if comm else dict(book=name, **{a: b for a, b in m.items()})
    row["corr_4x"] = round(corr_with(rn, REFn), 2)
    if L is not None:
        row["gross_bps"] = round((np.sign(L.w) * L.y).mean() * 1e4, 2); row["cost_bps"] = round(L.c.mean(), 2); row["t1%"] = round((L.tier == 0).mean() * 100, 1)
        if "entry" in L: row["x%"] = round((L.entry == "x1559").mean() * 100, 1)
    if extra: row.update(extra)
    rows.append(row); S[name] = rn; log("  " + " ".join(f"{a}={b}" for a, b in row.items()))
    return row
BOOKS = {}
for c in [0.3, 0.1, 1.0]:
    for pc, info, ent, nm in [("pR_o", "orig", "route", "ref_pR_o_rte"), ("pR_o", "orig", "moc", "ref_pR_o_moc"), ("o40s0", "orig", "route", "ref_o40s0_rte"),
                              ("pR_388", "59", "route", "diag_pR388_rte"), ("pR_378", "49", "moc", "a_pR378_moc"), ("p40_378", "49", "moc", "a_p40_378_moc"),
                              ("p26_378", "49", "moc", "a_p26_378_moc"), ("p40s0_378", "49", "moc", "a_p40s0_378_moc"), ("pR_388", "59", "x1559", "x_pR388_x1559")]:
        if c != 0.3 and nm not in ("ref_pR_o_rte", "a_pR378_moc", "a_p40_378_moc", "x_pR388_x1559"): continue
        if pc not in H0: continue
        H = prep(use_cut(H0, info), pc, ratios=())
        r, L = route_book(H, K, LAM, ent, "moo", c, 1.0, ret_pos=True)
        rec(f"{nm}|c{c}", r, L); BOOKS[(nm, c)] = (r, L)
        if nm in ("a_pR378_moc", "a_p40_378_moc"):     # lmt: LOC at the estimated 15:49 bid / ask on the (a) positions, selection re-run at this c
            for s in ([1.0, 1.2, 0.0] if c == 0.3 else [1.0]):
                P = lmt_exec(H, L, K, c=c, s=s); rl = lmt_daily(P, r.index); POS[(nm, c, s)] = P
                rec(f"lmt_{nm}_s{s}|c{c}", rl, None, extra=lmt_stats(P), comm=False)
        del H
    for nm, kw in [("b_two", {}), ("b3_two_m3", dict(mocslots=3))]:
        r, L = two_stage(H0, "pR_378", "pR_388", K, LAM, c, ret_pos=True, **kw); rec(f"{nm}|c{c}", r, L); BOOKS[(nm, c)] = (r, L)
    if c == 0.3:
        r, L = two_stage(H0, "p40_378", "p40_388", K, LAM, c, ret_pos=True); rec(f"b_two_p40|c{c}", r, L); BOOKS[("b_two_p40", c)] = (r, L)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/final6_{V}.csv", index=False); pd.set_option("display.width", 300)
log(f"--- final6 {V} ($0.4M, $3 per filled order; k6 lam3 MOO exit; c = auction coefficient, selection re-run at each c)\n" + T.to_string(index=False))
# correlations of the c=0.3 books
keys = [k for k in S if k.endswith("|c0.3") and not k.startswith(("lmt_a_p40", "a_p26", "a_p40s0"))]
M = pd.DataFrame({k: S[k] for k in keys}); M["onml4x_saved"] = REFn.reindex(M.index)
log("--- correlation (daily, $0.4M)\n" + M.corr().round(2).to_string())
# tiers and entry mix
for nm in ["ref_pR_o_rte", "a_pR378_moc", "b_two"]:
    L = BOOKS[(nm, 0.3)][1].copy(); L["y"] = L.y
    t = tier_table(L.assign(net=L.w * L.y - L.w.abs() * L.c / 1e4)); log(f"--- tiers {nm} c=0.3\n" + t.to_string(index=False))
P = POS.get(("a_pR378_moc", 0.3, 1.0))
if P is not None:
    tt = P.groupby("tier").apply(lambda x: pd.Series(lmt_stats(x))); log("--- lmt (a_pR378 positions, s=1) by tier\n" + tt.to_string())
    pd.to_pickle(P, f"{OUT}/lmt_pos_{V}.pkl")
pd.to_pickle({k: v for k, v in BOOKS.items() if k[1] == 0.3}, f"{OUT}/final6_{V}_books.pkl")
pd.to_pickle(S, f"{OUT}/final6_{V}_series.pkl")
if SAVE:
    note0 = ("per-day rank average of two quarterly LightGBM predictions (onml3 f40_on / f26calx feature sets, target yr, u2000) rebuilt on features with a {cut} "
             "information cutoff ({desc}); onml4 pipeline: causal yearly calibration E[y_oo|rank] - 3 x route cost, L/S 6+6 eq, MOO exit (c=0.3); ")
    SPEC = [("onml6_pR49_lam3_moc_moo_k6", ("a_pR378_moc", 0.3), "every entry a MOC order decided at 15:49:00 on the cut-378 prediction (features to the close of the 15:48 bar), auction c=0.3"),
            ("onml6_pR49x59_lam3_two_moo_k6", ("b_two", 0.3), "two-stage entry: at 15:49:00 the top 6/6 by the cut-378 score; names whose MOC cost <= expected cross cost (hs1559 from inputs to 15:49 + 1.85) are sent MOC; "
             "remaining slots filled at 15:59:00 with 15:59:30 crosses chosen on the cut-388 prediction (features to 15:59:00), cross cost hs1559 (inputs to 15:59:00) + 1.85 bps adverse selection"),
            ("onml6_p40n49_lam3_moc_moo_k6", ("a_p40_378_moc", 0.3), "as onml6_pR49_lam3_moc_moo_k6 but the f40 (liquidity-feature) prediction alone")]
    for name, key, desc in SPEC:
        if key not in BOOKS: continue
        r = BOOKS[key][0]
        rlib.save_cand("onml6", name, r, note0.format(cut="15:49 (and 15:59:00 for the cross leg)" if "two" in name else "15:49", desc=desc) + "every order uses only information available when it is sent; net of route cost + borrow 1%, BEFORE commission, decision-dated")
    P = POS.get(("a_pR378_moc", 0.3, 1.0))
    if P is not None:
        r = lmt_daily(P, BOOKS[("a_pR378_moc", 0.3)][0].index)
        rlib.save_cand("onml6", "onml6_pR49_lam3_locbid_moo_kSix", r, "positions of onml6_pR49_lam3_moc_moo_k6 (selection at 15:49:00); ENTRY = limit-on-close sent at 15:49:00 at the estimated 15:49 bid (short: ask) = c378 x (1 -+ hs49), hs49 = hs1559 booster on inputs to 15:49 "
                       "(real 15:49:00 NBBO check: proxy-vs-exact fill agreement 95-96%), filled iff the official close is strictly through it, fill at the official close + MOC impact c=0.3, unfilled = no trade; EXIT MOO c=0.3; "
                       "NET of impact, borrow 1% AND $3 per FILLED order on $0.4M (commission inside: do not charge again), decision-dated")
log(f"final6 {V} DONE")
