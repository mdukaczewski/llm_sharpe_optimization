#!/usr/bin/env python3
"""Build the lane frame (universe(600) cut of the 1500-name frame) and check that the old convention reproduces the saved candidate."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3lib import *
log(f"s0_frame start DEV_END={DEV_END}")
D = load_frame(600)
sig = cax(D); is150 = D["rank"] <= 150
P = positions(D[is150], sig[is150], k=10, mode="lo", smin=1.5)
r_old = net(P, old=True); ref = refs()["xsr_lo10_old"]
j = pd.concat([r_old.rename("new"), ref.rename("ref")], axis=1)
log(f"reproduction check old convention lo10_s15_u150: Sh new {sh(r_old):.3f} ref {sh(ref):.3f}  corr {j.new.corr(j.ref):.4f}  mean |diff| {(j.new-j.ref).abs().mean()*1e4:.3f} bps  rows {len(r_old)} / {len(ref)}")
log(f"y_oo vs y_on in the frame: corr {D.y_oo.corr(D.y_on):.3f}  mean bps {D.y_oo.mean()*1e4:.2f} / {D.y_on.mean()*1e4:.2f}  NaN share {D.y_oo.isna().mean():.4f} / {D.y_on.isna().mean():.4f}")
log(f"hs0931 median by rank tier: " + " ".join(f"{lo}-{hi}: {D.hs0931[(D['rank']>=lo)&(D['rank']<=hi)].median():.1f}" for lo, hi in [(1,50),(51,100),(101,150),(151,300),(301,600)]))
