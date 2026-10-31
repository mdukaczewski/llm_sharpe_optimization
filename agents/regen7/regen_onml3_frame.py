"""Rebuild agents/onml3 frame_u2000 with STRAT_END=2026-10-28 into the scratchpad (lib3.frame, unchanged code) and compare with the dev frame."""
import os, sys, time
assert os.environ.get("STRAT_END") == "2026-10-28"
sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml3")
import lib3
NEW = sys.argv[1]; OLD = lib3.OUT
lib3.OUT = NEW
D = lib3.frame(2000, rebuild=True)
print("new frame", len(D), D.date.min(), D.date.max(), flush=True)
