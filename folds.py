#!/usr/bin/env python3
"""Fold-based out-of-sample validation of the multi-sleeve portfolio.

The timeline is cut into half-year TEST windows (Jan-Jun, Jul-Dec; first one
starts at --first, default 2018-07-01).  For each fold every choice that was
made in-sample before -- which variant of each sleeve family (top-k, signal
threshold smin, exit bar, universe size, lookbacks, vol threshold, Thursday
skip, ...), which families to hold at all, and the sleeve weights -- is made
using ONLY dates before the test window (minus a 5-trading-day gap):

  --sel all : selection window = every prior date (train + val)
  --sel val : selection window = trailing 252 trading days (val only)

Per family the variant with the highest Sharpe over the selection window is
chosen (needs >= 200 non-zero days in the window).  Weights are inverse-vol x
max(Sharpe, 0), both from the selection window, so a family whose prior Sharpe
is negative gets weight 0.  The chosen variants and weights are then applied,
frozen, to the test window.  The test windows are concatenated and reported.

Each candidate series is itself causal (signals from past data; the ML sleeves
retrain on expanding windows), so a variant's return on day t never uses data
after t.  What this script removes is the choice-of-variant and choice-of-weight
look-ahead of portfolio.py.  It cannot remove the fact that the sleeve FAMILIES
and feature sets were designed after looking at the whole sample, nor the
survivorship in the minute-bar universe (names chosen by today's liquidity).

usage: python3 folds.py [--sel all|val] [--first 2018-07-01] [--gap 5]
"""
import argparse, glob, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
START, END = "2016-10-28", os.environ.get("FOLDS_END", "2026-10-28")
NEW_LANES = {"onml": ["onml", "onml2"], "dayml": ["dayml", "dayml2"], "xsrule": ["xsrule"], "etfmacro": ["etfmacro"]}   # family -> agent lanes (agents/<lane>/cands/*.csv); phase-4 lanes merge into their family
L = f"{HERE}/agents/intraday/logs"

def sh(r): return r.mean() / (r.std() + 1e-12) * np.sqrt(252)
def load(p):
    s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    return s[~s.index.duplicated()].sort_index()
def name(p): return os.path.basename(p).replace(".csv", "").replace("net_lgb1n+lgb5n+lgb10n_", "ml_")

def families():
    fam = {}
    fam["mom"] = {name(p): load(p) for p in sorted(glob.glob(f"{HERE}/logs/mom_grid/*.csv"))}
    vg = pd.read_csv(f"{HERE}/logs/vol_grid.csv", index_col=0, parse_dates=True)
    fam["vol"] = {c: vg[c] for c in vg.columns}
    fam["ml"] = {name(p): load(p) for p in sorted(glob.glob(f"{HERE}/agents/ml/logs/net_*.csv")) if "best" not in p}
    rule = []
    for tag in ["xs_all150", "xs_all150s1", "xs_all300s1", "xs_all600s1", "xs_all1200s1"]:
        rule += glob.glob(f"{L}/{tag}_XSLO*.csv") + glob.glob(f"{L}/{tag}_XSTOP*.csv") + glob.glob(f"{L}/{tag}_TS.csv") + glob.glob(f"{L}/{tag}_TSH.csv") + glob.glob(f"{L}/{tag}_XSLS.csv")
    fam["rule_e1"] = {name(p): load(p) for p in sorted(rule)}
    fam["rule_late"] = {name(p): load(p) for p in sorted(glob.glob(f"{L}/onml_x15_pred_rule_*.csv") + glob.glob(f"{L}/onml_x30_pred_rule_*.csv") + glob.glob(f"{L}/onml_x60_pred_rule_*.csv") + glob.glob(f"{L}/onml_a300x60_pred_rule_*.csv"))}
    fam["lgb_on"] = {name(p): load(p) for p in sorted(glob.glob(f"{L}/onml_*_pred_lgb_*.csv") + glob.glob(f"{L}/onml_*_pred_ridge_*.csv") + glob.glob(f"{L}/onml2_*_pred_*.csv"))}
    fam["intraday"] = {name(p): load(p) for p in sorted(glob.glob(f"{L}/dayml_*_pred_*.csv"))}
    for f_, lanes in NEW_LANES.items():
        d = {name(p): load(p) for lane in lanes for p in sorted(glob.glob(f"{HERE}/agents/{lane}/cands/*.csv"))}
        if d: fam[f_] = d
    return fam

def stats(r, label):
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    yr = r.groupby(r.index.year).apply(sh)
    print(f"{label:44s} Sharpe {sh(r):5.2f}  ret {r.mean()*252*100:5.1f}%  vol {r.std()*np.sqrt(252)*100:5.1f}%  maxDD {dd*100:6.1f}%  CAGR {eq.iloc[-1]**(252/len(r))*100-100:5.1f}%  yrs " + " ".join(f"{y}:{v:.1f}" for y, v in yr.items()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sel", default="all"); ap.add_argument("--first", default="2018-07-01"); ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--minsh", type=float, default=0.0, help="drop a family in a fold if its selected prior Sharpe is below this")
    ap.add_argument("--holdout", default="2026-04-28", help="first holdout date; dev folds end before it")
    ap.add_argument("--show_holdout", action="store_true", help="also evaluate the single holdout fold (look once, at the end)")
    ap.add_argument("--topn", type=int, default=1, help="average the top-n variants of a family (by prior Sharpe) instead of picking one")
    ap.add_argument("--wmode", default="shiv", help="family weights: shiv = max(Sharpe,0)/vol, iv = 1/vol if Sharpe>0, eq = equal, erc = equal risk contribution, mv = long-only min-variance (shrunk cov), ms = long-only max-Sharpe (shrunk cov, prior Sharpe)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--drop", default="", help="comma list of families to exclude")
    ap.add_argument("--keep", default="", help="regex: keep only candidate names matching it (coarser grid = less selection noise)")
    ap.add_argument("--omit", default="", help="regex: drop candidate names matching it")
    ap.add_argument("--voltarget", type=float, default=0.0, help="if >0: scale the portfolio to this annual vol using trailing-20d realized vol of its own past returns (capped at 2x)")
    ap.add_argument("--minhist", type=int, default=200, help="min non-zero days a variant needs in the selection window")
    ap.add_argument("--commission_bps", type=float, default=0.0, help="extra per-side cost (bps) charged on every fully-turned-over minute-bar book: L/S books pay 4x per day, long-only 2x (fixed per-order commission / clip)")
    ap.add_argument("--capital_usd", type=float, default=0.0, help="equity per book; with --order_usd charges each fully-turned-over minute-bar book n_positions x 2 orders x order_usd / capital per day (L/S top-k: 2k positions at 100%% long / 100%% short)")
    ap.add_argument("--order_usd", type=float, default=3.0, help="fixed commission per filled order (used with --capital_usd)")
    a = ap.parse_args()
    fam = families()
    if a.capital_usd > 0:
        import re as _re
        def npos(f, n):
            if f in ("mom", "vol", "ml", "etfmacro"): return 0                 # monthly / multi-day / 1-2 ETF orders: negligible
            m = _re.search(r"(?:top|_k|lo|ls|XSLO|LO|LS)(\d+)", n); k = int(m.group(1)) if m else 20
            sides = 1 if ("_LO_" in n or "_lo" in n or "XSLO" in n or n.endswith("_LO") or "LO_top" in n) else 2
            return k * sides
        fam = {f: {n: s - npos(f, n) * 2 * a.order_usd / a.capital_usd for n, s in d.items()} for f, d in fam.items()}
        print(f"commission: ${a.order_usd:.0f} per order on ${a.capital_usd/1e6:.2f}M equity per book (L/S top-20: {40*2*a.order_usd/a.capital_usd*1e4:.1f} bps/day)")
    elif a.commission_bps > 0:
        def gross_of(f, n):
            if f in ("mom", "vol", "ml"): return 0.0                       # monthly / multi-day books: commission negligible
            if f == "etfmacro": return 0.0                                 # one ETF order per side on half the book: $3 is < 0.2 bp
            if "_LO_" in n or "_lo" in n or "XSLO" in n or "_LO" in n: return 1.0
            return 2.0                                                     # L/S top-k books
        fam = {f: {n: s - 2 * gross_of(f, n) * a.commission_bps / 1e4 for n, s in d.items()} for f, d in fam.items()}
        print(f"commission: {a.commission_bps} bps per side on minute-bar books")
    for f_ in [x for x in a.drop.split(",") if x]: fam.pop(f_, None)
    import re
    if a.keep: fam = {f: {n: s for n, s in d.items() if re.search(a.keep, n)} for f, d in fam.items()}
    if a.omit: fam = {f: {n: s for n, s in d.items() if not re.search(a.omit, n)} for f, d in fam.items()}
    fam = {f: d for f, d in fam.items() if d}
    allser = {f"{f}:{n}": s for f, d in fam.items() for n, s in d.items()}
    df = pd.DataFrame(allser).loc[START:END].fillna(0.0)
    dates = df.index
    print("families: " + ", ".join(f"{f}({len(d)})" for f, d in fam.items()) + f"   days {len(dates)}   sel={a.sel}")
    # test windows
    starts = pd.date_range(a.first, END, freq="6MS"); starts = starts[starts.month.isin([1, 7])] if len(starts) else starts
    hold = pd.Timestamp(a.holdout); folds = []
    for k, ts in enumerate(starts):
        if ts >= hold: break
        te = (starts[k + 1] - pd.Timedelta(days=1)) if k + 1 < len(starts) else pd.Timestamp(END)
        folds.append((ts, min(te, hold - pd.Timedelta(days=1))))
    if a.show_holdout: folds.append((hold, pd.Timestamp(END)))
    test_parts, rows, picks = [], [], {f: [] for f in fam}
    for ts, te in folds:
        pos = int(np.searchsorted(dates, ts)); sel_end = pos - a.gap
        sel_start = 0 if a.sel == "all" else max(0, sel_end - 252)
        sel = df.iloc[sel_start:sel_end]; test = df.loc[ts:te]
        if len(test) == 0: continue
        wts = {}; chosen = {}; fser_sel = {}; fser_test = {}
        for f, d in fam.items():
            scored = []
            for n in d:
                s = sel[f"{f}:{n}"]; s = s[s != 0]
                if len(s) < a.minhist: continue
                scored.append((sh(s), n))
            if not scored: continue
            scored.sort(reverse=True); top = [n for _, n in scored[:a.topn]]
            cols = [f"{f}:{n}" for n in top]
            fs = sel[cols].mean(axis=1); ft = test[cols].mean(axis=1); bsh = sh(fs)
            if bsh < a.minsh or bsh <= 0: continue
            chosen[f] = ("+".join(top) if a.topn > 1 else top[0], bsh); fser_sel[f] = fs; fser_test[f] = ft
            wts[f] = bsh / (fs.std() + 1e-12) if a.wmode == "shiv" else (1.0 / (fs.std() + 1e-12) if a.wmode == "iv" else 1.0)
        test = pd.DataFrame(fser_test); sel_f = pd.DataFrame(fser_sel)
        if a.wmode in ("erc", "mv", "ms") and len(wts) > 1:
            C = sel_f.cov().values * 252; n_ = len(wts); names_ = list(sel_f.columns)
            if a.wmode == "erc":
                x = np.ones(n_) / n_
                for _ in range(500):                      # cyclical ERC fixed point
                    mrc = C @ x; x = (1.0 / np.maximum(mrc, 1e-12)) ; x = x / x.sum()
                    x = x * np.sqrt(np.ones(n_) / np.maximum(np.diag(C), 1e-12)) ** 0; x = x / x.sum()
            else:
                lam = 0.5; Cs = (1 - lam) * C + lam * np.diag(np.diag(C))     # shrink off-diagonals
                mu = np.ones(n_) if a.wmode == "mv" else np.array([chosen[f][1] for f in names_]) * np.sqrt(np.diag(Cs))
                x = np.linalg.solve(Cs, mu); x = np.clip(x, 0, None); x = x / (x.sum() + 1e-12)
            wts = {f: float(x[i]) for i, f in enumerate(names_)}
        w = pd.Series(wts); w = w / w.sum()
        r = (test[w.index] * w).sum(axis=1)
        if a.voltarget > 0:
            hist_r = (sel_f[w.index] * w).sum(axis=1)
            full = pd.concat([hist_r, r]); rv = full.rolling(20).std().shift(1) * np.sqrt(252)
            r = r * (a.voltarget / rv.reindex(r.index)).clip(upper=2.0).fillna(1.0)
        test_parts.append(r)
        rows.append(dict(test=f"{ts.date()}..{te.date()}", days=len(test), fold_sharpe=round(sh(r), 2), **{f: f"{chosen[f][0]} ({chosen[f][1]:.1f}, w={w.get(f, 0):.2f})" if f in chosen else "-" for f in fam}))
        for f in fam: picks[f].append(test[f] if f in chosen else r * 0)
    oos = pd.concat(test_parts); tab = pd.DataFrame(rows)
    if a.show_holdout:
        ho = test_parts[-1]; oos = pd.concat(test_parts[:-1]); tab = tab.iloc[:-1]
    if not a.quiet:
        pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80); print(tab.to_string(index=False)); print()
        for f in fam: stats(pd.concat(picks[f]), f"  OOS stitched family {f}")
    label = f"sel={a.sel} topn={a.topn} w={a.wmode}"
    stats(oos, f"DEV FOLDS OOS [{label}]")
    if a.show_holdout: stats(ho, f"HOLDOUT {a.holdout}..{END} [{label}]")
    if not a.quiet: print(f"fold Sharpes: mean {tab.fold_sharpe.mean():.2f}  median {tab.fold_sharpe.median():.2f}  min {tab.fold_sharpe.min():.2f}  negative folds {(tab.fold_sharpe<0).sum()}/{len(tab)}")
    # reference: the in-sample fixed portfolio.py MAIN over the same test span
    try:
        if a.quiet: raise RuntimeError("quiet")
        pe = pd.read_csv(f"{HERE}/out/portfolio_equity.csv", index_col=0, parse_dates=True)["main_ret"].loc[oos.index[0]:oos.index[-1]]
        stats(pe, "reference: portfolio.py MAIN (in-sample picks), same span")
    except Exception as e: print("no reference:", e)
    os.makedirs(f"{HERE}/out", exist_ok=True)
    if not a.quiet: oos.rename("ret").to_csv(f"{HERE}/out/folds_{a.sel}_oos.csv", index_label="date"); tab.to_csv(f"{HERE}/out/folds_{a.sel}_table.csv", index=False)
if __name__ == "__main__": main()
