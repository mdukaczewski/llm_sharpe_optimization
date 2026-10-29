#!/usr/bin/env python3
"""dayml FINAL: regenerates ONLY the kept candidates end-to-end from raw minute data.
Honours STRAT_END through rlib (rlib.DEV_END): run once with STRAT_END=2026-10-28 for the holdout.
usage: nohup python3 agents/dayml/final.py  (universe(1500), LightGBM n_jobs=4, runs alone)
KEPT is filled in at the end of the research session (see REPORT.md)."""
import sys, time, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import dlib, rlib, numpy as np, pandas as pd
NJ = int(os.environ.get("DAYML_NJOBS", "4")); LANE = "dayml"
print("DEV_END", rlib.DEV_END, flush=True)
KEPT = []   # (cand_name, feature-set name, target, retrain, k, weight, min_names, note)
