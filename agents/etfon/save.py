"""Save the etfon candidates from x1_res.parquet (run x1_ts.py first). Series are decision-dated (rlib.book convention) and
NET of 1.5 bps/side (MOC entry + MOO exit), 1% borrow on shorts AND the $3/order commission at $0.4M (2 x positions x 0.075 bp/day)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, elib2, rlib
H = os.path.dirname(os.path.abspath(__file__)); R = pd.read_parquet(f"{H}/x1_res.parquet")
C = "COMMISSION INSIDE ($3/order at $0.4M, 2 orders per position per day; do NOT charge again); 1.5 bps/side MOC+MOO, borrow 1%; decision-dated; "
rlib.save_cand("etfon", "etfon_ts_s1_moo", R["eq_oo_c1.5_s1.0_lo"], C + "16 equity/sector ETFs, long o/n if s=0.5z30-z60>1, w=0.05/rv20(t-1) cap .5; ~2.9 pos/day; corr xsr3 0.77; NOT recommended (marginal contribution negative)")
rlib.save_cand("etfon", "etfon_ts_s15_both_moo", R["eq_oo_c1.5_s1.5_ls"], C + "same, long s>1.5 / short s<-1.5; ~3.0 pos/day; corr xsr3 0.48; NOT recommended")
print(open(f"{H}/cands/NOTES.txt").read())
