"""Final: build the chosen candidates, tables (full / fold-OOS / halves / maxDD / orders / cost split / correlations),
save with rlib.save_cand.  usage: python s5_final.py [save]"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dml2")
from s2_port import *
from refs import corrs, refs
SAVE = "save" in sys.argv
NOTE = ("EXECUTION: all trades at the official close (MOC) of day t, decided on information up to the close of t-1 (order can be "
        "sent any time before the 15:50 MOC cutoff of t); PnL of holdings set at close t is booked on t+1 (exit-dated); "
        "COSTS inside the series: per side closing-auction impact liq.auction_cost_bps(order_usd, max(exp(dvc), 0.05*exp(ldv20)), "
        "vol20_daily, c=0.3) with a 1 bp floor (3 bps for names outside the liq universe), $3 per order actually sent "
        "(names whose holding changes that day, incl. SPY hedge) on $0.4M equity, borrow 1%/yr on shorts. folds.py must "
        "charge NO extra commission. Book: LightGBM daily-bar models (agents/ml features, 5-day and 10-day close-to-close "
        "targets, retrained every 126 days, expanding window), score = 5-day mean of the rank-averaged predictions, "
        "ADV20 >= $50M, equal-weight L/S 100%/100% of equity, ")
CANDS = {   # name: (tags, sim kwargs, description)
    "dml2_b5b10_band_n15_r10_b20_H": (["b5_r126", "b10_r126"], dict(k=15, mode="band", reb=10, buf=20),
        "15 long / 15 short, re-ranked every 10 trading days, a holding is kept while its rank stays within top/bottom 35 (enter top 15, exit beyond 35), SPY hedge of the trailing-252d beta (cap 0.5) at MOC"),
    "dml2_b5b10_band_n15_r5_b20_H": (["b5_r126", "b10_r126"], dict(k=15, mode="band", reb=5, buf=20),
        "15 long / 15 short, re-ranked weekly (every 5 trading days), enter top 15, exit beyond 35, SPY hedge (cap 0.5) at MOC"),
    "dml2_b5b10_band_n20_r10_b45_H": (["b5_r126", "b10_r126"], dict(k=20, mode="band", reb=10, buf=45),
        "20 long / 20 short, re-ranked every 10 trading days, enter top 20, exit beyond 65, SPY hedge (cap 0.5) at MOC"),
    "dml2_b5b10_band_n12_r1_b10_H": (["b5_r126", "b10_r126"], dict(k=12, mode="band", reb=1, buf=10),
        "12 long / 12 short, re-ranked daily, enter top 12, exit beyond 22, SPY hedge (cap 0.5) at MOC"),
}
EXTRA = {}
if os.path.exists(f"{OUT}/final_extra.py"): exec(open(f"{OUT}/final_extra.py").read())
CANDS.update(EXTRA)

if __name__ == "__main__":
    P, F, m = setup(); aud = liq_mats(P); sim = Sim(P, F, m, aud); rows = []; series = {}
    old = pd.read_csv(f"{OUT}/sim_orig_A.csv", index_col=0, parse_dates=True) if os.path.exists(f"{OUT}/sim_orig_A.csv") else None
    for name, (tags, kw, desc) in CANDS.items():
        S = score(P, m, tags, 5); df = sim.run(S, **kw); r = summ(df, name)
        for c in [0.1, 1.0]: r[f"sh_c{c}"] = round(rlib.sh(sim.run(S, c=c, **kw).net), 2)
        r.update(corrs(df.net)); r["corrSPY"] = round(pd.concat([df.net, pd.Series(F["spy_ret"], P.dates)], axis=1, join="inner").corr().iloc[0, 1], 2)
        yr = df.net.groupby(df.net.index.year).apply(rlib.sh); r["years"] = " ".join(f"{y%100}:{v:+.1f}" for y, v in yr.items())
        rows.append(r); series[name] = df.net
        if SAVE:
            rlib.save_cand("dml2", name, df.net, NOTE + desc + f". Orders/day {r['ord_d']}, impact {r['imp_bp']} bps/day, commission {r['com_bp']} bps/day.")
            df.to_csv(f"{OUT}/sim_{name}.csv", float_format="%.7g")
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/final_table.csv", index=False)
    print(R.drop(columns=["years"]).to_string()); print(R[["name", "years"]].to_string())
    oos, tab = rlib.fold_oos(series, label="dml2 family"); print(tab.to_string())
    C = pd.DataFrame(series).corr().round(2); print(C.to_string())
    if old is not None: print("corr with re-costed original (MOC, $3):", {k: round(pd.concat([v, old.net], axis=1).corr().iloc[0, 1], 2) for k, v in series.items()})
