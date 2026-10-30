#!/usr/bin/env python3
"""lmt step 3b (overnight): the limit-on-close-at-the-bid entry ('pbid') in detail.
(a) exact check on the panel days: onml4x positions whose ENTRY day t is a panel day, real 15:59:30 bid / ask vs the
    c388 x (1 -/+ hs1559) proxy (fill = official close strictly through the limit).
(b) sensitivity: proxy scale s, strict / touch, auction coefficient c, and posting to more names (K = 6 / 9 / 12 per side,
    each filled position sized as in the k6 book: w = 1/6, order $33k) so that the expected number of fills is ~ 6.
(c) passive exit vs MOO at c = 1.0.  Tiers, years, correlation with onml4x_pR_lam3_rte_moo_k6, family fold-OOS."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/lmt")
from books_on import *
REF = pd.read_csv(f"{R}/agents/onml4x/cands/onml4x_pR_lam3_rte_moo_k6.csv", index_col=0, parse_dates=True).iloc[:, 0] - 24 * 3 / CAP

def sized(P, k_size=6):
    P = P.copy(); P["w"] = np.sign(P.w) / k_size; return P

def tier_tab(P):
    out = []
    for t in range(4):
        x = P[P.tier == t]
        if len(x) == 0: continue
        xf = x[x.fill]; sg = np.sign(xf.w)
        out.append(f"t{t}: {len(x)/len(P)*100:.0f}% of posts, fill {x.fill.mean()*100:.0f}%, net {((sg*xf.ret) - xf.cost/1e4).mean()*1e4:+.1f} bps")
    return " | ".join(out)

if __name__ == "__main__":
    H = setup(); log("=== books_on2 (pbid detail)"); pd.set_option("display.width", 250)
    # ---- (a) exact on panel days
    Q = pd.read_csv(f"{R}/out/quotes_panel.csv.gz"); Q = Q[(Q.date <= rlib.DEV_END) & (Q.t == 1559)].dropna(subset=["bid", "ask"]); Q["date"] = pd.to_datetime(Q.date)
    r0, L = route_book(H, 6, 3.0, "route", "moo", 0.3, 1.0, ret_pos=True)
    P = execute(H, L, 6, "pbid", "moo"); P0 = execute(H, L, 6, "route", "moo")
    X = P.reset_index().merge(Q[["tk", "date", "bid", "ask"]], on=["tk", "date"], how="inner").set_index("index")
    cl = H.loc[X.index, "c_off"].values; lg = X.w.values > 0
    for rule in ["strict", "touch"]:
        fe = np.where(lg, (cl < X.bid.values) if rule == "strict" else (cl <= X.bid.values), (cl > X.ask.values) if rule == "strict" else (cl >= X.ask.values))
        Lp = np.where(lg, lim(H.loc[X.index, "c388"].values, H.loc[X.index, "hs1559"].values, "buy"), lim(H.loc[X.index, "c388"].values, H.loc[X.index, "hs1559"].values, "sell"))
        fp = np.where(lg, (cl < Lp) if rule == "strict" else (cl <= Lp), (cl > Lp) if rule == "strict" else (cl >= Lp))
        net = (np.sign(X.w) * X.ret - X.cost / 1e4).values * 1e4; net0 = (np.sign(P0.loc[X.index, "w"]) * P0.loc[X.index, "ret"] - P0.loc[X.index, "cost"] / 1e4).values * 1e4
        log(f"(a) panel entry days, n {len(X)} [{rule}]: exact fill {fe.mean()*100:.1f}% proxy {fp.mean()*100:.1f}% agree {(fe == fp).mean()*100:.1f}% | "
            f"net bps (MOC+MOO) exact-filled {net[fe].mean():+.2f} (se {net[fe].std()/np.sqrt(fe.sum()):.1f}) exact-unfilled {net[~fe].mean():+.2f} | proxy-filled {net[fp].mean():+.2f} unfilled {net[~fp].mean():+.2f} | "
            f"route book on the same rows {net0.mean():+.2f}")
        # exact-based daily PnL on panel days: per position share of all signals
        log(f"     per signal (unfilled = 0): exact {np.where(fe, net, 0).mean():+.2f} proxy {np.where(fp, net, 0).mean():+.2f} vs route book {net0.mean():+.2f} bps")
    # ---- (b) sensitivity
    rows = []; S = {}
    def add(name, P, dates, k_size=6, extra=None):
        r = daily(P, k_size, dates=dates); S[name] = r; m = metrics(r); pt = per_trade(P)
        nf = P[P.fill].groupby("date").size().reindex(dates).fillna(0)
        row = dict(name=name, **pt, fills_day=round(nf.mean(), 1), **m, corr_ref=round(pd.concat([r, REF], axis=1).dropna().corr().iloc[0, 1], 2))
        if extra: row.update(extra)
        rows.append(row); log(str({a: b for a, b in row.items() if a != "yrs"})); return r
    dates = r0.index
    add("ref route+moo (c=0.3)", P0, dates)
    for rule in ["strict", "touch"]:
        for s in [0.75, 1.0, 1.25]:
            Pp = execute(H, L, 6, "pbid", "moo", rule=rule, s_close=s); add(f"pbid moo K6 s{s} {rule}", Pp, dates)
    for c in [0.1, 1.0]:
        add(f"ref route+moo c={c}", execute(H, L, 6, "route", "moo", c=c), dates)
        add(f"pbid moo K6 strict c={c}", execute(H, L, 6, "pbid", "moo", c=c), dates)
        for ex in ["p0932w30", "p0931w30"]:
            add(f"route {ex} strict c={c}", execute(H, L, 6, "route", ex, c=c), dates)
            add(f"pbid {ex} strict c={c}", execute(H, L, 6, "pbid", ex, c=c), dates)
    pos = {}
    for K in [9, 12]:
        rK, LK = route_book(H, K, 3.0, "route", "moo", 0.3, 1.0, ret_pos=True)
        for rule in ["strict", "touch"]:
            Pp = sized(execute(H, LK, 6, "pbid", "moo", rule=rule)); pos[(K, rule)] = Pp
            add(f"pbid moo post K{K} size k6 {rule}", Pp, dates, extra=dict(tiers=tier_tab(Pp)))
        for c in [0.1, 1.0]:
            add(f"pbid moo post K{K} size k6 strict c={c}", sized(execute(H, LK, 6, "pbid", "moo", c=c)), dates)
    Pp = execute(H, L, 6, "pbid", "moo"); rows[-1]["tiers"] = rows[-1].get("tiers")
    log("tiers ref:  " + tier_tab(P0)); log("tiers pbid K6: " + tier_tab(Pp)); log("tiers pbid K12: " + tier_tab(pos[(12, "strict")]))
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/books_on2.csv", index=False); pd.to_pickle(S, f"{OUT}/books_on2_series.pkl")
    log("--- books_on2\n" + T.drop(columns=["yrs", "tiers"], errors="ignore").to_string(index=False))
    log("--- years\n" + T[["name", "yrs"]].to_string(index=False))
    # family / blends (decision-dated, all net of commission)
    fam = {"ref": REF.loc[:rlib.DEV_END], "pbidK6": S["pbid moo K6 s1.0 strict"], "pbidK12": S["pbid moo post K12 size k6 strict"]}
    for a_, b_ in [("ref", "pbidK6"), ("ref", "pbidK12")]:
        fo, tab = rlib.fold_oos({a_: fam[a_], b_: fam[b_]}, p=False); log(f"family fold-OOS {{{a_}, {b_}}}: {rlib.sh(fo):.2f} picks {tab.pick.value_counts().to_dict()}")
    log("books_on2 DONE")
