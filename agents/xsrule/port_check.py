"""Orientation only: inverse-vol portfolio of the 7 MAIN sleeves with / without each xsrule candidate; cross-correlations."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule")
from xlib import *
MAIN = ["mom_longonly", "vol_ens2_net", "ml_ls", "xslo20s1_150", "onrule_x60_lo20", "onlgb1200q_ls20", "daylgb1200_ls20"]
S = {m: pd.read_csv(f"/home/user/llm_sharpe_optimization/sleeves/{m}.csv", index_col=0, parse_dates=True).iloc[:, 0] for m in MAIN}
C = rlib.load_cands("xsrule"); R = refs()
df = pd.DataFrame({**S, **C}).loc["2017-01-03":DEV_END].fillna(0.0)
def iv(cols, start="2017-01-03"):
    x = df.loc[start:, cols]; w = 1 / x.std(); w = w / w.sum(); return (x * w).sum(1)
base = iv(MAIN); print(f"MAIN 7 sleeves inverse-vol: Sh {sh(base):.2f}  (fold-OOS of the stitched series n/a; orientation only)")
for c in C:
    p = iv(MAIN + [c]); p2 = iv([m for m in MAIN if m != "xslo20s1_150"] + [c])
    print(f"  + {c:24s} Sh {sh(p):.2f} | replacing xslo20s1_150: {sh(p2):.2f} | corr to MAIN port {df[c].corr(base):.2f}")
p = iv(MAIN + ["xsr_cAx_lo20_s15_u150", "xsr_cBx_ls20_u1500"]); print(f"  + cAx LO150 + cBx LS1500: {sh(p):.2f}")
p = iv([m for m in MAIN if m != "xslo20s1_150"] + ["xsr_cAx_lo20_s15_u150", "xsr_cAx_ls20_u150"]); print(f"  replace rule sleeve by cAx LO150, + cAx LS150: {sh(p):.2f}")
print("\ncorrelations (candidates x sleeves):")
print(df[list(C)].corrwith(df["xslo20s1_150"]).rename("xslo20s1_150").to_frame().join(df[list(C)].corrwith(df["onlgb1200q_ls20"]).rename("onlgb1200q")).join(df[list(C)].corrwith(df["daylgb1200_ls20"]).rename("daylgb1200")).join(df[list(C)].corrwith(R["spy_cc"].reindex(df.index)).rename("spy_cc")).round(2))
print("\ncandidate x candidate:"); print(df[list(C)].corr().round(2))
