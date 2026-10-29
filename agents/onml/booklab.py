"""Commission-aware book lab over the saved 3-seed overnight predictions (P positions x $6 / $400k per day)."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
CAP = 400000.0; t0 = time.time(); D = load(1500); D = D[D.date <= rlib.DEV_END].reset_index(drop=True)
P = pd.read_parquet(f"{HERE}/agents/onml/out/pred_f28cal_q3.parquet"); M = D.merge(P, on=["date", "tk"], how="left"); pred = M["p"]
rows = []
for mode_lo in [False, True]:
    for k in [6, 8, 10, 12, 15, 20]:
        for w in ["eq", "pred"]:
            if mode_lo:
                r = rlib.book(M, pred, k=k, y="y_on", mode="lo", weight=w, p=False); npos = k
            else:
                r = book2(M, pred, k=k, weight=w, p=False); npos = 2 * k
            rn = r - npos * 6 / CAP; fo = rlib.fold_oos(rn, p=False)[0]
            rows.append(("lo" if mode_lo else "ls", k, w, round(rlib.sh(r), 2), round(rlib.sh(rn), 2), round(rlib.sh(fo), 2), f"{rn.mean()*252*100:.0f}%"))
T = pd.DataFrame(rows, columns=["mode", "k", "w", "Sh_spread", "Sh_$0.4M", "foldOOS_$0.4M", "ret_$0.4M"])
pd.set_option("display.width", 200); print(T.sort_values("foldOOS_$0.4M", ascending=False).to_string(index=False)); T.to_csv(f"{HERE}/agents/onml/booklab.csv", index=False)
print("LAB DONE", flush=True)
