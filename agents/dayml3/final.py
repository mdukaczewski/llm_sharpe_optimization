#!/usr/bin/env python3
"""dayml3 FINAL (books from the saved quarterly predictions out/pred_final_<cfg>_Q.parquet; fit them with fitq.py first).
Scoring modes:
  sel   : gross-target model(s) (A1 [+ A1x120 pair, average per-day pct rank]) with the CAUSAL cost-aware two-sided selection
          long score E[y_30d | rank] - 0.5 c, short -E - 0.5 c  (c = predicted round-trip half-spread, calibration on prior dates)
  pair  : two-sided net-target models (long B1, short S1)
Candidates (<= 8): dayml3_<desc>_x05_k{4,6,8} (f = 0.5) and dayml3_<desc>_x1_k{6,8} (f = 1); series = gross - f x spread cost - borrow,
BEFORE the $3 commission.   python3 agents/dayml3/final.py                       (env D3_SEL_CFGS='A1,A1x120'  D3_UNI=u600)"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
SEL = [c for c in os.environ.get("D3_SEL_CFGS", "A1").split(",") if c]; CSV = os.environ.get("D3_CSV", "final.csv"); UNI = os.environ.get("D3_UNI", "u600"); PAIR = os.environ.get("D3_PAIR", "")      # e.g. "B1,S1"
log(f"==== final: sel models {SEL} universe {UNI} pair {PAIR or '-'} ====")
D = frame(1500); u600 = (D["rank"] <= 600).to_numpy(); u300 = (D["rank"] <= 300).to_numpy(); UNIS = {"u1500": None, "u600": u600, "u300": u300}
def load(cfg):
    P = pd.read_parquet(f"{OUT}/pred_final_{cfg}_Q.parquet"); return D[["date", "tk"]].merge(P, on=["date", "tk"], how="left")["p"].to_numpy(float)
preds = {c: load(c) for c in SEL + ([x for x in PAIR.split(",")] if PAIR else [])}
for c, p in preds.items():
    icm, ict = rlib.ic(D, pd.Series(p, index=D.index), "y_30d"); log(f"final {c}: coverage {np.isfinite(p).mean():.3f} IC(y_30d) {icm:.4f} t {ict:.1f}")
scores = {}
for c in SEL: scores[f"sel_{c}"] = sel_scores(D, preds[c], 0.5)
if len(SEL) > 1:
    e = np.mean(np.column_stack([xr(D, preds[c]) for c in SEL]), axis=1); scores["sel_ens"] = sel_scores(D, e, 0.5)
    scores["gross_ens"] = (e, -e)
if SEL: scores[f"gross_{SEL[0]}"] = (xr(D, preds[SEL[0]]), -xr(D, preds[SEL[0]]))
if PAIR: b_, s_ = PAIR.split(","); scores["pair"] = (preds[b_], preds[s_])
rows = []; books = {}
for tag, (ls, ss) in scores.items():
    for uni, m in UNIS.items():
        for k in [2, 3, 4, 6, 8]:
            L = positions(D, None, k=k, y="y_30d", mask=m, long_score=ls, short_score=ss); dd = daily(L, k); books[(tag, uni, k)] = (L, dd)
            for f in [1.0, 0.5, 0.25]:
                s = stats(net(dd, f)); s.update(cfg=tag, uni=uni, k=k, f=f); rows.append(s)
            s = stats(net(dd, old=True)); s.update(cfg=tag, uni=uni, k=k, f="old"); rows.append(s)
            s = stats(net(dd, 0.0)); s.update(cfg=tag, uni=uni, k=k, f="gross-comm"); rows.append(s)
            t = tier_table(L, 0.5); sh_ = " ".join(f"{r.tier}:{r.share:.2f}/{r.net_bps:+.1f}" for r in t.itertuples())
            log(f"{tag:10s} {uni:5s} k{k} | f1 Sh {rows[-5]['sh']:5.2f} fo {rows[-5]['fo']:5.2f} | f0.5 Sh {rows[-4]['sh']:5.2f} fo {rows[-4]['fo']:5.2f} h {rows[-4]['h1']:4.1f}/{rows[-4]['h2']:4.1f} mdd {rows[-4]['mdd']*100:5.0f}% | f0.25 fo {rows[-3]['fo']:5.2f} | old Sh {rows[-2]['sh']:5.2f} fo {rows[-2]['fo']:5.2f} | tiers {sh_}")
            dd.to_parquet(f"{OUT}/final_{tag}_{uni}_k{k}.parquet")
pd.DataFrame(rows).to_csv(f"{OUT}/{CSV}", index=False)
# ---- candidates
ref = pd.read_csv(f"{R}/agents/dayml2/cands/dayml2_ens_e30_k8.csv", index_col=0, parse_dates=True).iloc[:, 0]
sl = pd.read_csv(f"{R}/sleeves/onlgb1200q_ls20.csv", index_col=0, parse_dates=True).iloc[:, 0].loc[:DEV_END]
spy = rlib.ticker_frame("SPY")["y_cc"].shift(1).loc[:DEV_END]
tag = os.environ.get("D3_SAVE_TAG", "sel_ens" if len(SEL) > 1 else (f"sel_{SEL[0]}" if SEL else "pair")); desc = os.environ.get("D3_DESC", "liqsel") + ("" if UNI == "u1500" else f"_{UNI}")
for k in [4, 6, 8]:
    L, dd = books[(tag, UNI, k)]
    for f, xt in [(0.5, "x05"), (1.0, "x1")]:
        if xt == "x1" and (k == 4 or os.environ.get("D3_X1", "1") != "1"): continue
        r = net(dd, f, comm=False); rc = net(dd, f); s = stats(rc); name = f"dayml3_{desc}_{xt}_k{k}"
        cc = pd.DataFrame({"x": rc, "ref": ref, "on": sl, "spy": spy}).dropna().corr()["x"]
        t = tier_table(L, f); tiers = " ".join(f"{r_.tier}:{r_.share:.2f}/{r_.net_bps:+.1f}bps" for r_ in t.itertuples())
        note = (f"intraday L/S top/bottom {k} eq, decide 09:31, fill 09:33 bar (bin 3), exit 10:01 bar (bin 31); {tag} = quarterly LightGBM {SEL if tag != 'pair' else PAIR} on dlib.EXT + 17 liq features, "
                + ("causal cost-aware selection (E[y_30d|rank] - 0.5c long / -E - 0.5c short); " if tag != "pair" else "two-sided net targets rank(y_60d - 0.5c) long / rank(-y_60d - 0.5c) short, top-k of each; ")
                + f" universe {UNI}; series NET of f={f} x predicted NBBO half-spread (liq.py hs0931 entry + hs1001 exit) + borrow, BEFORE commission | "
                f"$0.4M: Sh {s['sh']:.2f} fold-OOS {s['fo']:.2f} halves {s['h1']:.2f}/{s['h2']:.2f} mdd {s['mdd']*100:.0f}% | corr dayml2_ens_e30_k8 {cc['ref']:.2f} onlgb {cc['on']:.2f} SPY {cc['spy']:.2f} | tiers {tiers}"
                + ("" if s['fo'] > 0 else " | NEGATIVE at this spread assumption: not tradable"))
        save(name, r, note); log(f"saved {name}: {note}")
log("final done")
