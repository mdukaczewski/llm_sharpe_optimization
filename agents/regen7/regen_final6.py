"""final6.py Q, reduced to the 5 saved candidates (same frame construction, same book calls, same order of use_cut state changes).
No metrics are computed; series written in the rlib.save_cand format (NOTES.txt not touched)."""
import os, sys, time
assert os.environ.get("STRAT_END") == "2026-10-28" or os.environ.get("REGEN_TEST") == "1"
OUTD = sys.argv[1]
sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml6")
from lmt6 import *
import glob
K = 6; LAM = 3.0; O3 = f"{HERE}/agents/onml3/out"
def say(*a): print(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a), flush=True)
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
    say(f"{kcol} <- {[os.path.basename(f) for f in fs]}")
H0["hs1559_orig"] = H0.hs1559
for c_ in H0.columns:
    if H0[c_].dtype == np.float64: H0[c_] = H0[c_].astype(np.float32)
def pr(a, b): return prank((H0[a] + H0[b]) / 2, H0.date)
H0["pR_o"] = pr("o40", "o26"); H0["pR_378"] = pr("p40_378", "p26_378"); H0["pR_388"] = pr("p40_388", "p26_388")
pcols = [c for c in H0.columns if c.startswith(("o40", "o26", "p40", "p26", "pR_"))]
H0 = H0[H0[pcols].notna().all(1)].reset_index(drop=True)
say(f"H0 {len(H0)} rows {H0.date.min().date()}..{H0.date.max().date()} pcols {pcols}")
def save(name, r):
    r = r.loc[rlib.START:rlib.DEV_END].rename("ret").rename_axis("date")
    r.to_csv(f"{OUTD}/{name}.csv", float_format="%.8f"); say(f"saved {name} {len(r)} rows {r.index[0].date()}..{r.index[-1].date()} NaN {int(r.isna().sum())}")
c = 0.3
# final6 loop order at c=0.3: ..., diag_pR388 (59), a_pR378 (49), a_p40_378 (49), ...  (use_cut sets H0 in place)
use_cut(H0, "59")
out = {}
for pc, info, ent, nm in [("pR_378", "49", "moc", "a_pR378_moc"), ("p40_378", "49", "moc", "a_p40_378_moc")]:
    H = prep(use_cut(H0, info), pc, ratios=())
    r, L = route_book(H, K, LAM, ent, "moo", c, 1.0, ret_pos=True)
    P = lmt_exec(H, L, K, c=c, s=1.0); rl = lmt_daily(P, r.index)
    out[nm] = (r, rl); del H
use_cut(H0, "59")          # state after x_pR388_x1559, before two_stage (two_stage reads only the *_49 / *_59 columns)
rb, Lb = two_stage(H0, "pR_378", "pR_388", K, LAM, c, ret_pos=True)
save("onml6_pR49_lam3_moc_moo_k6", out["a_pR378_moc"][0])
save("onml6_pR49x59_lam3_two_moo_k6", rb)
save("onml6_p40c49_lam3_moc_moo_k6", out["a_p40_378_moc"][0])
save("onml6_pRcut_lamThree_locbid_moo_kSix", out["a_pR378_moc"][1])
save("onml6_pForty_lamThree_locbid_moo_kSix", out["a_p40_378_moc"][1])
say("FINAL6 REGEN DONE")
