#!/usr/bin/env python3
"""lmt step 3 (intraday): passive entry / exit on the dayml3 intraday book.
Signal: agents/dayml3/out/pred_final_A1_Q.parquet (quarterly LightGBM, gross target, liq features; decision at the end of the
09:31 bar = 09:32:00).  Positions: per day top / bottom k by the prediction within ranks <= U (no cost-aware selection unless
--sel), equal weight 1/k per position (unfilled positions stay in cash: no rescaling).
Execution:
  x1 / x05 / mid : dayml3 convention, fill 09:33 bar close (bin 3), exit 10:01 bar close (bin 31), paying f x predicted
                   half-spread (hs0931 entry, hs1001 exit)
  pW{W}          : the user's rule: at 09:32:00 post a buy limit at the bid (short: sell at the ask), proxy limit = 09:31-bar
                   close x (1 -/+ s hs0931); strict fill within W bars (bins 2..W+1); filled -> exit 10:01 bar close crossing
                   (sell at the bid: pays hs1001); unfilled = no trade
  pW{W}fb        : unfilled -> cross at the close of the last window bar (pays hs0931 / hs0935), then as above
  pW{W}px{W2}    : filled entries exit passively: sell limit at the ask posted 10:01:00 (mid = 10:00-bar close, hs1001), strict
                   fill within W2 bars, else cross at the close of the last window bar (hs1001)
Series exit-dated = decision-dated (same day).  Net of spread, borrow 1%/yr and $3 per filled order."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/lmt")
from lmtlib import *

def frame():
    E = ext(); P = pd.read_parquet(f"{R}/agents/dayml3/out/pred_final_A1_Q.parquet"); P["date"] = pd.to_datetime(P.date)
    D = E[~E.half].merge(P, on=["date", "tk"], how="inner")
    D["rank"] = D.tk.map(rlib.RANK); D["tier"] = np.where(D["rank"] <= 300, 0, np.where(D["rank"] <= 600, 1, 2))
    D["y"] = D.c31 / D.c3 - 1; D.loc[D.y.abs() > 0.5, "y"] = np.nan
    for c in ["hs0931", "hs0935", "hs1001"]:
        D[c] = D[c].fillna(D.groupby(["date", "tier"])[c].transform("median"))
    return D[D.y.notna() & D.p.notna() & D.c1.notna()].reset_index(drop=True)

def positions(D, k, U, score=None):
    g = D[D["rank"] <= U].copy(); s = g.p if score is None else score.loc[g.index]
    g["s"] = s; g["n"] = g.groupby("date").s.transform("size"); g = g[g.n >= 2 * k + 5]
    g["rl"] = g.groupby("date").s.rank(method="first", ascending=False); g["rs"] = g.groupby("date").s.rank(method="first")
    top = g[g.rl <= k].copy(); top["w"] = 1.0 / k; bot = g[g.rs <= k].copy(); bot["w"] = -1.0 / k
    return pd.concat([top, bot])

FBE = {2: ("c3", "hs0931"), 4: ("c5", "hs0935"), 30: ("c31", "hs1001")}
FBX = {4: ("c34", "hs1001"), 30: ("c60", "hs1001")}
def execute(L, mode, s=1.0, rule="strict", f=1.0):
    lg = (L.w > 0).values; n = len(L); fill = np.ones(n, bool)
    if mode in ("x1", "x05", "mid"):
        ff = {"x1": 1.0, "x05": 0.5, "mid": 0.0}[mode] * f
        ret = L.y.values; cost = ff * (L.hs0931.values + L.hs1001.values)
    else:
        W = int(mode[2:].split("fb")[0].split("px")[0]); Le = np.where(lg, lim(L.c1.values, L.hs0931.values, "buy", s), lim(L.c1.values, L.hs0931.values, "sell", s))
        fe = np.where(lg, filled(L[f"lo_0932_{W}"].values, L[f"hi_0932_{W}"].values, Le, "buy", rule), filled(L[f"lo_0932_{W}"].values, L[f"hi_0932_{W}"].values, Le, "sell", rule))
        px_in = Le.astype(float); e_cost = np.zeros(n)
        if "fb" in mode:
            c_, h_ = FBE[W]; px_in = np.where(fe, Le, L[c_].values); e_cost = np.where(fe, 0.0, f * L[h_].values); fill = np.ones(n, bool)
        else:
            fill = fe
        if "px" in mode:
            W2 = int(mode.split("px")[1]); Lx = np.where(lg, lim(L.c30.values, L.hs1001.values, "sell", s), lim(L.c30.values, L.hs1001.values, "buy", s))
            fx = np.where(lg, filled(L[f"lo_1001_{W2}"].values, L[f"hi_1001_{W2}"].values, Lx, "sell", rule), filled(L[f"lo_1001_{W2}"].values, L[f"hi_1001_{W2}"].values, Lx, "buy", rule))
            c_, h_ = FBX[W2]; px_out = np.where(fx, Lx, L[c_].values); x_cost = np.where(fx, 0.0, f * L[h_].values)
        else:
            px_out = L.c31.values; x_cost = f * L.hs1001.values
        ret = px_out / px_in - 1; cost = e_cost + x_cost
    P = pd.DataFrame({"date": L.date.values, "tk": L.tk.values, "tier": L.tier.values, "w": L.w.values, "ret": ret, "cost": cost, "y": L.y.values}, index=L.index)
    P["fill"] = fill & np.isfinite(ret) & np.isfinite(cost); P["norders"] = 2 * P.fill.astype(int)
    return P

def tiers(P):
    out = []
    for t in [0, 1, 2]:
        x = P[(P.tier == t)]; xf = x[x.fill]; sg = np.sign(xf.w)
        out.append(f"t{t}:{len(x)/len(P)*100:.0f}%/fill{x.fill.mean()*100:.0f}%/net{((sg*xf.ret) - xf.cost/1e4).mean()*1e4:+.1f}")
    return " ".join(out)

if __name__ == "__main__":
    D = frame(); log(f"=== books_day: rows {len(D)} days {D.date.nunique()}")
    dates = pd.DatetimeIndex(sorted(D.date.unique())); res = []; S = {}
    for U in [300, 600, 1500]:
        for k in [4, 8]:
            L = positions(D, k, U)
            modes = ["mid", "x1", "x05", "pW2", "pW4", "pW30", "pW2fb", "pW4fb", "pW4px4", "pW4px30", "pW30px30", "pW4fbpx30"]
            for mode in modes:
                for rule in (["strict", "touch"] if mode.startswith("p") else ["strict"]):
                    P = execute(L, mode, rule=rule); r = daily(P, k, dates=dates); nm = f"u{U}|k{k}|{mode}|{rule}"; S[nm] = r
                    # adverse selection: mid-to-mid gross (09:33 -> 10:01 bar closes) of filled vs unfilled positions
                    sg = np.sign(P.w); ag = (sg * P.y)
                    row = dict(U=U, k=k, mode=mode, rule=rule, **per_trade(P), y_filled=round(ag[P.fill].mean() * 1e4, 2), y_unfilled=round(ag[~P.fill].mean() * 1e4, 2) if (~P.fill).any() else np.nan,
                               y_all=round(ag.mean() * 1e4, 2), **metrics(r), tiers=tiers(P))
                    res.append(row); log(str({a: b for a, b in row.items() if a != "yrs"}))
    T = pd.DataFrame(res); T.to_csv(f"{OUT}/books_day.csv", index=False); pd.to_pickle(S, f"{OUT}/books_day_series.pkl")
    pd.set_option("display.width", 300); log("--- books_day\n" + T.drop(columns=["yrs"]).to_string(index=False)); log("books_day DONE")
