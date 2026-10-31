import os, sys
assert os.environ.get("STRAT_END") == "2026-10-28"
sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml6")
import lib6
lib6.OUT = sys.argv[1]
lib6.build()
print("CUT BUILD DONE", flush=True)
