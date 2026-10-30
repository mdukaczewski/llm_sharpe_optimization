#!/usr/bin/env python3
"""Books for one prediction vintage.  usage: python3 explore6.py Y   (yearly seed-0 predictions)  |  Q (quarterly)
Original (official-close / 15:59 features) predictions: Y -> onml3 pred_y_f40_on + pred_y_f26x_on (yearly seed 0, same protocol);
Q -> onml3 pred_f40_on + pred_f26calx_q3 (quarterly 3-seed, the onml4x inputs) and, when present, the 1-seed pred_f40_on_q_s0."""
from book6 import *
V = sys.argv[1]; K = 6; LAM = 3.0
O3 = f"{HERE}/agents/onml3/out"
if V == "Y":
    preds = {"o40": f"{O3}/pred_y_f40_on.parquet", "o26": f"{O3}/pred_y_f26x_on.parquet"}
    for c in (378, 388):
        for s in ("f40", "f26"): preds[f"{s[1:]}_{c}"] = f"{OUT}/pred_{s}_c{c}_Y_s0.parquet"
    preds = {("o40" if k == "o40" else "o26" if k == "o26" else ("p" + k)): v for k, v in preds.items()}
else:
    preds = {"o40": f"{O3}/pred_f40_on.parquet", "o26": f"{O3}/pred_f26calx_q3.parquet"}
    for c in (378, 388):
        for s in ("f40", "f26"):
            fs = sorted(__import__("glob").glob(f"{OUT}/pred_{s}_c{c}_Q_s*.parquet"))
            if fs: preds[f"p{s[1:]}_{c}"] = fs     # several seeds -> rank-averaged below
H0 = load_H({k: v for k, v in preds.items() if isinstance(v, str)})
for k, v in preds.items():
    if isinstance(v, list):
        acc = None
        for i, f in enumerate(v):
            P = pd.read_parquet(f, columns=["date", "tk", "p"]).rename(columns={"p": f"_s{i}"}); H0 = H0.merge(P, on=["date", "tk"], how="left")
        H0[k] = prank(H0[[f"_s{i}" for i in range(len(v))]].mean(1, skipna=False), H0.date); H0 = H0.drop(columns=[f"_s{i}" for i in range(len(v))])
        log(f"explore6 {V}: {k} = rank average of {len(v)} seeds")
H0["hs1559_orig"] = H0.hs1559
def pr(a, b): return prank((H0[a] + H0[b]) / 2, H0.date)
H0["pR_o"] = pr("o40", "o26")
for c in (378, 388):
    if f"p40_{c}" in H0 and f"p26_{c}" in H0: H0[f"pR_{c}"] = pr(f"p40_{c}", f"p26_{c}")
# common rows: names with every prediction (so books differ only by the information used)
pcols = [c for c in H0.columns if c.startswith(("o40", "o26", "p40_", "p26_", "pR_"))]
H0 = H0[H0[pcols].notna().all(1)].reset_index(drop=True)
log(f"explore6 {V}: {len(H0)} rows {H0.date.min().date()}..{H0.date.max().date()}; preds {pcols}")
for p in pcols: log(f"  IC y_oo {p}: {rlib.ic(H0, H0[p], 'y_oo')[0]:.4f}")
REF = ref_series()
rows = []; S = {}
def rec(name, r, L=None, extra=None):
    m = full_metrics(r, K); row = mrow(name, m); row["corr_4x"] = round(corr_with(commission(r, K), commission(REF, K)), 2)
    if L is not None:
        row["gross_bps"] = round((np.sign(L.w) * L.y).mean() * 1e4, 2); row["cost_bps"] = round(L.c.mean(), 2); row["t1%"] = round((L.tier == 0).mean() * 100, 1)
        if "entry" in L: row["x%"] = round((L.entry == "x1559").mean() * 100, 1)
    if extra: row.update(extra)
    rows.append(row); S[name] = r; log("  " + " ".join(f"{a}={b}" for a, b in row.items()))
for c in [0.3, 0.1, 1.0]:
    for pc, info in [("pR_o", "orig"), ("pR_388", "59"), ("pR_378", "49"), ("o40", "orig"), ("p40_378", "49"), ("o26", "orig"), ("p26_378", "49")]:
        if pc not in H0: continue
        if c != 0.3 and pc not in ("pR_o", "pR_378", "pR_388"): continue
        H = prep(use_cut(H0, info), pc, ratios=())
        for ent in (["route", "moc"] + (["x1559"] if info == "59" else [])):
            r, L = route_book(H, K, LAM, ent, "moo", c, 1.0, ret_pos=True)
            rec(f"{pc}|{ent}|c{c}", r, L)
    if "pR_378" in H0 and "pR_388" in H0:
        for st in ["all"]:
            r, L = two_stage(H0, "pR_378", "pR_388", K, LAM, c, ret_pos=True, stage2=st); rec(f"two(pR_378,pR_388)|{st}|c{c}", r, L)
        if c == 0.3:
            r, L = two_stage(H0, "pR_378", "pR_o", K, LAM, c, ret_pos=True); rec(f"two(pR_378,pR_o: x leg on official-close preds)|c{c}", r, L)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/explore6_{V}.csv", index=False)
pd.set_option("display.width", 250); log(f"--- explore6 {V} ($0.4M, $3/order, k6 lam3, MOO exit; route = per-name MOC or 15:59:30 cross at hs1559 + 1.85)\n" + T.to_string(index=False))
pd.to_pickle(S, f"{OUT}/explore6_{V}_series.pkl")
log(f"explore6 {V} DONE")
