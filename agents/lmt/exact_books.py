#!/usr/bin/env python3
"""lmt step 2b: EXACT passive fills for the books' OWN positions on the NBBO-panel days (real bid / ask at T from
out/quotes_panel.csv.gz, minute-bar lows / highs from out/panel_bars.npz), against the minute-bar proxy on the same rows.
(i)  overnight onml4x_pR_lam3_rte_moo_k6 positions whose EXIT day t+1 is a panel day:
       passive exit at 09:31:00 at the real ask (long) / bid (short), strict fill within W bars (bins 1..W);
       unfilled -> cross at the real 09:35:00 (W = 4) / 10:01:00 (W = 30) bid (long) / ask (short).
       vs MOO and vs crossing at the real 09:31:00 quote.  Prices relative to the official open of t+1 (then x (1 + y_oo)).
(ii) intraday dayml3 A1 top/bottom-k positions (u600, k8) on panel days: passive entry at 09:35:00 at the real bid
       (long) / ask (short), strict fill within bins 5..30, exit crossing at the real 10:01:00 quote; vs crossing at 09:35.
Proxy on the same rows: limit = previous bar close x (1 -/+ predicted hs); fallback / exit prices = bar close -/+ predicted hs."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/lmt")
from lmtlib import *
Z = np.load(f"{OUT}/panel_bars.npz", allow_pickle=True); PB = list(Z["bins"]); col = {b: i for i, b in enumerate(PB)}
idx = pd.DataFrame({"tk": Z["tk"], "date": pd.to_datetime(Z["dates"])}); idx["row"] = np.arange(len(idx)); HB, LB, CB = Z["h"], Z["l"], Z["c"]
Q = pd.read_csv(f"{R}/out/quotes_panel.csv.gz"); Q = Q[Q.date <= rlib.DEV_END]; Q["date"] = pd.to_datetime(Q.date)
Qw = Q.pivot_table(index=["tk", "date"], columns="t", values=["bid", "ask"]); Qw.columns = [f"{a}{b}" for a, b in Qw.columns]; Qw = Qw.reset_index()
for t in [931, 935, 1001]:
    bad = ~((Qw[f"ask{t}"] >= Qw[f"bid{t}"]) & (Qw[f"bid{t}"] > 0) & ((Qw[f"ask{t}"] - Qw[f"bid{t}"]) / Qw[f"bid{t}"] < 0.2)); Qw.loc[bad, [f"bid{t}", f"ask{t}"]] = np.nan
idx = idx.merge(Qw, on=["tk", "date"], how="left")
E = ext()[["date", "tk", "hs0931", "hs0935", "hs1001", "o_off"]]; idx = idx.merge(E, on=["date", "tk"], how="left")

def win(rows, b0, W, side):
    a = (LB if side == "buy" else HB)[rows][:, [col[b] for b in range(b0, b0 + W)]]
    return np.nanmin(a, 1) if side == "buy" else np.nanmax(a, 1)

def run_on():
    import books_on as bo
    H = bo.setup(); r0, L = bo.route_book(H, 6, 3.0, "route", "moo", 0.3, 1.0, ret_pos=True)
    h = H.loc[L.index, ["date", "tk", "y_oo", "n_date", "r_moc"]].copy(); h["w"] = L.w.values; h["entry"] = L.entry.values
    C = bo.rc(H.loc[L.index], 6); h["e_cost"] = np.where(h.entry == "x1559", C["x1559"][1].values, C["moc"][1].values); h["moo"] = C["moo"][1].values
    h["e_adj"] = np.where(h.entry == "x1559", h.r_moc.fillna(0), 0.0)
    X = h.merge(idx.rename(columns={"date": "n_date"}), on=["tk", "n_date"], how="inner"); X = X[X.bid931.notna() & X.o_off.notna()]
    rows = X.row.values; lg = (X.w > 0).values; oo = X.o_off.values; g = (1 + X.y_oo.values) * (1 + X.e_adj.values)
    out = []
    def rec(name, ratio, xcost, xfill=None):
        net = np.sign(X.w.values) * (g * ratio - 1) * 1e4 - X.e_cost.values - xcost
        out.append(dict(variant=name, n=len(X), exit_fill=(np.nan if xfill is None else round(xfill.mean() * 100, 1)), net_bps=round(net.mean(), 2), se=round(net.std() / np.sqrt(len(X)), 2),
                        net_long=round(net[lg].mean(), 2), net_short=round(net[~lg].mean(), 2)))
    rec("MOO (model)", np.ones(len(X)), X.moo.values)
    bid1, ask1 = X.bid931.values, X.ask931.values
    rec("cross 09:31:00 real quote", np.where(lg, bid1, ask1) / oo, np.zeros(len(X)))
    rec("cross 09:31 bar proxy (c1 -/+ hs0931)", CB[rows, col[1]] / oo, X.hs0931.values)
    for W, fb in [(4, 935), (30, 1001)]:
        for rule in ["strict", "touch"]:
            Lx = np.where(lg, ask1, bid1); hi = win(rows, 1, W, "sell"); lo = win(rows, 1, W, "buy")
            fx = np.where(lg, (hi > Lx) if rule == "strict" else (hi >= Lx), (lo < Lx) if rule == "strict" else (lo <= Lx))
            fbp = np.where(lg, X[f"bid{fb}"].values, X[f"ask{fb}"].values)
            ratio = np.where(fx, Lx, fbp) / oo; rec(f"passive 09:31 real, W{W}, fb real {fb} [{rule}]", ratio, np.zeros(len(X)), fx)
            # proxy on the same rows
            hc = "hs0935" if fb == 935 else "hs1001"; mh = CB[rows, col[0]]
            Lp = np.where(lg, lim(mh, X.hs0931.values, "sell"), lim(mh, X.hs0931.values, "buy"))
            fp = np.where(lg, (hi > Lp) if rule == "strict" else (hi >= Lp), (lo < Lp) if rule == "strict" else (lo <= Lp))
            fbb = W  # fallback at the close of the last window bar, paying the predicted hs
            ratio = np.where(fp, Lp, CB[rows, col[fbb]]) / oo; rec(f"passive 09:31 PROXY, W{W} [{rule}]", ratio, np.where(fp, 0.0, X[hc].values), fp)
            if rule == "strict": log(f"   on-exit W{W}: exact fill {fx.mean()*100:.1f}% proxy {fp.mean()*100:.1f}% agreement {(fx == fp).mean()*100:.1f}%")
    T = pd.DataFrame(out); log("--- exact vs proxy, onml4x_pR_lam3_rte_moo_k6 positions with a panel exit day (bps per position, net of entry + exit cost, before commission)\n" + T.to_string(index=False))
    return T

def run_day():
    import books_day as bd
    D = bd.frame(); L = bd.positions(D, 8, 600)
    X = L.merge(idx, on=["tk", "date"], how="inner", suffixes=("", "_q")); X = X[X.bid935.notna() & X.bid1001.notna()]
    rows = X.row.values; lg = (X.w > 0).values; sg = np.sign(X.w.values); out = []
    def rec(name, pin, pout, cost, fill=None):
        net = sg * (pout / pin - 1) * 1e4 - cost; f = np.ones(len(X), bool) if fill is None else fill
        out.append(dict(variant=name, n=len(X), fill=round(f.mean() * 100, 1), net_filled=round(net[f].mean(), 2), per_signal=round(np.where(f, net, 0).mean(), 2)))
    m935 = (X.bid935.values + X.ask935.values) / 2; m1001 = (X.bid1001.values + X.ask1001.values) / 2
    xin = np.where(lg, X.ask935.values, X.bid935.values); xout = np.where(lg, X.bid1001.values, X.ask1001.values)
    rec("mid 09:35 -> mid 10:01 (real)", m935, m1001, 0)
    rec("cross 09:35 -> cross 10:01 (real)", xin, xout, 0)
    rec("bar proxy: c4 + hs0935 -> c31 - hs1001", CB[rows, col[4]], CB[rows, col[31]], X.hs0935.values + X.hs1001.values)
    for rule in ["strict", "touch"]:
        Le = np.where(lg, X.bid935.values, X.ask935.values); lo = win(rows, 5, 26, "buy"); hi = win(rows, 5, 26, "sell")
        fe = np.where(lg, (lo < Le) if rule == "strict" else (lo <= Le), (hi > Le) if rule == "strict" else (hi >= Le))
        rec(f"passive 09:35 real, bins 5-30, exit cross 10:01 real [{rule}]", Le, xout, 0, fe)
        mh = CB[rows, col[4]]; Lp = np.where(lg, lim(mh, X.hs0935.values, "buy"), lim(mh, X.hs0935.values, "sell"))
        fp = np.where(lg, (lo < Lp) if rule == "strict" else (lo <= Lp), (hi > Lp) if rule == "strict" else (hi >= Lp))
        rec(f"passive 09:35 PROXY, exit c31 - hs1001 [{rule}]", Lp, CB[rows, col[31]], X.hs1001.values, fp)
        mm = sg * (m1001 / m935 - 1) * 1e4
        log(f"   day [{rule}] exact fill {fe.mean()*100:.1f}% proxy {fp.mean()*100:.1f}% agree {(fe == fp).mean()*100:.1f}% | mid->mid gross of filled {mm[fe].mean():+.2f} unfilled {mm[~fe].mean():+.2f} all {mm.mean():+.2f} bps")
    T = pd.DataFrame(out); log("--- exact vs proxy, dayml3 A1 u600 k8 positions on panel days (bps)\n" + T.to_string(index=False))
    return T

if __name__ == "__main__":
    log("=== exact_books"); pd.set_option("display.width", 250)
    a = run_on(); b = run_day(); a.to_csv(f"{OUT}/exact_on.csv", index=False); b.to_csv(f"{OUT}/exact_day.csv", index=False); log("exact_books DONE")
