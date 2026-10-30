#!/usr/bin/env python3
"""lmt step 1: EXACT fill study on the NBBO quote-panel days (every 40th trading day, all 2000 names, dates <= DEV_END).
A resting buy limit posted at time T at the real NBBO bid B_T (sell: at the ask A_T) is
  strict-filled within W minutes if some trade prints STRICTLY through the limit (minute-bar low < B; high > A for a sell) in
  bins b0 .. b0+W-1  (bin b covers [09:30+b, 09:31+b); T = 09:31:00 -> b0 = 1, 09:35:00 -> 5, 10:01:00 -> 31).  A minute
  bar's low/high is the extreme eligible trade price in that minute, so this is the tick rule "a trade at a price better than
  my limit happened" at 1-minute time resolution; queue position is unknown, so 'strict' is the conservative rule and
  'touch' (low <= B) the optimistic upper bound.
Closing auction: a buy limit at the 15:59:30 bid that rests into the close fills iff official close < B (strict) / <= B
  (touch), at the official close price (the auction price), continuous fills in the last 30 s ignored.
Adverse selection: return from the fill price to a later mark (a later NBBO mid of the panel, or a bar close) for filled
  orders vs the return from mid_T to the same mark for filled / unfilled / all.
Proxy (for the full history, step 2): limit = mid_hat x (1 -/+ s x predicted half-spread), mid_hat = close of the minute bar
  before T (09:30 bar for 09:31:00), compared with the exact fill on the same rows."""
import os, sys, numpy as np, pandas as pd
R = "/home/user/llm_sharpe_optimization"; sys.path.insert(0, R)
import rlib
OUT = f"{R}/agents/lmt/out"; LOG = f"{R}/agents/lmt/log.txt"
def log(s):
    print(s, flush=True); open(LOG, "a").write(s + "\n")
Z = np.load(f"{OUT}/panel_bars.npz", allow_pickle=True); PB = list(Z["bins"]); col = {b: i for i, b in enumerate(PB)}
idx = pd.DataFrame({"tk": Z["tk"], "date": Z["dates"]}); H, Lw, C = Z["h"], Z["l"], Z["c"]
Q = pd.read_csv(f"{R}/out/quotes_panel.csv.gz"); Q = Q[(Q.date <= rlib.DEV_END) & ~Q.tk.isin(["VIXY", "UVXY", "VIXM", "SPY", "QQQ"])]
Qw = Q.pivot_table(index=["tk", "date"], columns="t", values=["bid", "ask"]); Qw.columns = [f"{a}{b}" for a, b in Qw.columns]; Qw = Qw.reset_index()
P = idx.merge(Qw, on=["tk", "date"], how="left")
for t in [931, 935, 1001, 1559]:
    ok = (P[f"ask{t}"] >= P[f"bid{t}"]) & (P[f"bid{t}"] > 0) & ((P[f"ask{t}"] - P[f"bid{t}"]) / P[f"bid{t}"] < 0.2)
    P.loc[~ok, [f"bid{t}", f"ask{t}"]] = np.nan
    P[f"mid{t}"] = (P[f"bid{t}"] + P[f"ask{t}"]) / 2; P[f"hs{t}r"] = (P[f"ask{t}"] - P[f"bid{t}"]) / (2 * P[f"mid{t}"]) * 1e4
P["rank"] = P.tk.map(rlib.RANK); P["tier"] = pd.cut(P["rank"], [0, 300, 600, 1500, 2000], labels=["1-300", "301-600", "601-1500", "1501-2000"])
E = pd.read_parquet(f"{OUT}/ext.parquet", columns=["date", "tk", "hs0931", "hs0935", "hs1001", "hs1559"]); E["date"] = E.date.dt.strftime("%Y-%m-%d")
P = P.merge(E, on=["date", "tk"], how="left")
G = pd.read_parquet(f"{R}/agents/onml4/out/G.parquet", columns=["date", "tk", "y_oo", "y_on"]); G["date"] = G.date.dt.strftime("%Y-%m-%d")
P = P.merge(G, on=["date", "tk"], how="left")
P["close"] = Z["close"]
log(f"=== study1: panel rows {len(P)} days {P.date.nunique()} valid 09:31 quotes {P.bid931.notna().sum()}")

def first_fill(lim, b0, side, rule, Wmax=60):
    """first W (1..Wmax) at which the order is filled, 0 if never; lim per row"""
    out = np.zeros(len(lim), int)
    for W in range(Wmax, 0, -1):
        b = b0 + W - 1
        if b not in col: continue
        x = (Lw if side == "buy" else H)[:, col[b]]
        if side == "buy": f = (x < lim) if rule == "strict" else (x <= lim + 1e-6)
        else: f = (x > lim) if rule == "strict" else (x >= lim - 1e-6)
        out[f & np.isfinite(lim)] = W
    return out

POST = {"0931": (931, 1, 0, "hs0931"), "0935": (935, 5, 4, "hs0935"), "1001": (1001, 31, 30, "hs1001")}
rows = []; adv = []; prox = []
for nm, (t, b0, pb, hcol) in POST.items():
    v = P[f"bid{t}"].notna().to_numpy()
    for side in ["buy", "sell"]:
        lim = (P[f"bid{t}"] if side == "buy" else P[f"ask{t}"]).to_numpy(float)
        for rule in ["strict", "touch"]:
            ff = first_fill(lim, b0, side, rule); P[f"ff_{nm}_{side}_{rule}"] = ff
            for W in [1, 2, 4, 10, 30, 60]:
                if b0 + W - 1 not in col: continue
                fl = (ff > 0) & (ff <= W)
                for tr in ["all"] + list(P.tier.cat.categories):
                    m = v & ((P.tier == tr).to_numpy() if tr != "all" else True)
                    rows.append(dict(T=nm, side=side, rule=rule, W=W, tier=tr, n=int(m.sum()), fill=round(fl[m].mean() * 100, 1),
                                     hs_real=round(P.loc[m, f"hs{t}r"].median(), 1)))
    # adverse selection: (post, window, mark)
    marks = {"0931": [(4, "mid935"), (30, "mid1001")], "0935": [(26, "mid1001")], "1001": [(30, "c60")]}[nm]
    for W, mk in marks:
        mark = P[mk].to_numpy(float) if mk.startswith("mid") else C[:, col[60]].astype(float)
        mid = P[f"mid{t}"].to_numpy(float); hs = P[f"hs{t}r"].to_numpy(float)
        for side in ["buy", "sell"]:
            sg = 1 if side == "buy" else -1
            lim = (P[f"bid{t}"] if side == "buy" else P[f"ask{t}"]).to_numpy(float)
            rmid = sg * (mark / mid - 1) * 1e4                          # mid_T -> mark, in the order's direction
            rfill = sg * (mark / lim - 1) * 1e4                         # fill price -> mark
            for rule in ["strict", "touch"]:
                ff = P[f"ff_{nm}_{side}_{rule}"].to_numpy(); fl = (ff > 0) & (ff <= W)
                for tr in ["all"] + list(P.tier.cat.categories):
                    m = np.isfinite(rmid) & np.isfinite(mark) & ((P.tier == tr).to_numpy() if tr != "all" else True)
                    adv.append(dict(T=nm, W=W, mark=mk, side=side, rule=rule, tier=tr, n=int(m.sum()), fill=round(fl[m].mean() * 100, 1), hs=round(np.nanmean(hs[m]), 1),
                                    mid_all=round(np.nanmean(rmid[m]), 2), mid_filled=round(np.nanmean(rmid[m & fl]), 2), mid_unfilled=round(np.nanmean(rmid[m & ~fl]), 2),
                                    fill_to_mark=round(np.nanmean(rfill[m & fl]), 2), cross_all=round(np.nanmean(rmid[m] - hs[m]), 2),
                                    passive_per_signal=round(np.nansum(rfill[m & fl]) / m.sum(), 2)))
    # proxy agreement (strict rule, buy and sell) for W in (4, 30)
    for s in [0.5, 0.75, 1.0, 1.25]:
        for side in ["buy", "sell"]:
            sg = -1 if side == "buy" else 1
            mh = C[:, col[pb]].astype(float)
            lp = np.round(mh * (1 + sg * s * P[hcol].to_numpy(float) / 1e4), 2)
            ffp = first_fill(lp, b0, side, "strict"); ffr = P[f"ff_{nm}_{side}_strict"].to_numpy()
            for W in [4, 30]:
                m = v & np.isfinite(lp)
                a = (ffr > 0) & (ffr <= W); b = (ffp > 0) & (ffp <= W)
                mark = (P["mid935"] if (nm == "0931" and W == 4) else (P["mid1001"] if nm != "1001" else pd.Series(C[:, col[60]]))).to_numpy(float)
                if nm == "0935" and W == 30: continue
                lim = (P[f"bid{t}"] if side == "buy" else P[f"ask{t}"]).to_numpy(float); ssg = 1 if side == "buy" else -1
                rr = ssg * (mark / lim - 1) * 1e4; rp = ssg * (mark / lp - 1) * 1e4; mm = m & np.isfinite(mark)
                prox.append(dict(T=nm, side=side, s=s, W=W, n=int(mm.sum()), fill_exact=round(a[mm].mean() * 100, 1), fill_proxy=round(b[mm].mean() * 100, 1),
                                 agree=round((a == b)[mm].mean() * 100, 1), lim_err_bps=round(np.nanmedian(ssg * (lp[mm] / lim[mm] - 1) * 1e4), 2),
                                 ret_exact=round(np.nanmean(rr[mm & a]), 2), ret_proxy=round(np.nanmean(rp[mm & b]), 2),
                                 per_sig_exact=round(np.nansum(rr[mm & a]) / mm.sum(), 2), per_sig_proxy=round(np.nansum(rp[mm & b]) / mm.sum(), 2)))
# closing auction
t = 1559; cl = P.close.to_numpy(float); v = P.bid1559.notna().to_numpy() & np.isfinite(cl)
for side in ["buy", "sell"]:
    sg = 1 if side == "buy" else -1; lim = (P.bid1559 if side == "buy" else P.ask1559).to_numpy(float)
    for rule in ["strict", "touch"]:
        if side == "buy": fl = (cl < lim) if rule == "strict" else (cl <= lim + 1e-6)
        else: fl = (cl > lim) if rule == "strict" else (cl >= lim - 1e-6)
        yoo = sg * P.y_oo.to_numpy(float) * 1e4; rmoc = sg * (cl / P.mid1559.to_numpy(float) - 1) * 1e4
        for tr in ["all"] + list(P.tier.cat.categories):
            m = v & np.isfinite(yoo) & ((P.tier == tr).to_numpy() if tr != "all" else True)
            adv.append(dict(T="1559auction", W=0, mark="y_oo", side=side, rule=rule, tier=tr, n=int(m.sum()), fill=round(fl[m].mean() * 100, 1), hs=round(P.loc[m, "hs1559r"].mean(), 1),
                            mid_all=round(np.nanmean(yoo[m]), 2), mid_filled=round(np.nanmean(yoo[m & fl]), 2), mid_unfilled=round(np.nanmean(yoo[m & ~fl]), 2),
                            fill_to_mark=round(np.nanmean(yoo[m & fl]), 2), cross_all=round(np.nanmean(rmoc[m]), 2), passive_per_signal=round(np.nansum(yoo[m & fl]) / m.sum(), 2)))
A = pd.DataFrame(rows); B = pd.DataFrame(adv); X = pd.DataFrame(prox)
A.to_csv(f"{OUT}/s1_fill.csv", index=False); B.to_csv(f"{OUT}/s1_adverse.csv", index=False); X.to_csv(f"{OUT}/s1_proxy.csv", index=False)
P.drop(columns=[c for c in P.columns if c.startswith("ff_")]).head(0)
P[["tk", "date", "tier"] + [c for c in P.columns if c.startswith("ff_")]].to_parquet(f"{OUT}/s1_ff.parquet")
pd.set_option("display.width", 250)
log("--- fill rates (%) at the real NBBO, strict = trade through the limit, by tier\n" +
    A[A.W.isin([1, 4, 30])].pivot_table(index=["T", "side", "rule", "W"], columns="tier", values="fill").to_string())
log("--- adverse selection, bps in the order's direction (buy: + = price up).  mid_* = mid_T -> mark; fill_to_mark = limit price -> mark (filled only); "
    "cross_all = crossing at T for every signal; passive_per_signal = filled PnL averaged over ALL signals (unfilled = 0)\n" + B.to_string(index=False))
log("--- proxy limit (bar close before T x (1 -/+ s x predicted hs)) vs exact, strict rule\n" + X.to_string(index=False))
log("study1 DONE")
