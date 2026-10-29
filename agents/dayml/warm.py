import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml"); import dlib, rlib
tks = rlib.universe(1500); t0 = time.time(); n = 0
for tk in tks:
    try: dlib.ticker_rows(tk); n += 1
    except Exception as e: print(tk, "skip", e, flush=True)
    if n % 100 == 0: print(n, f"{time.time()-t0:.0f}s", flush=True)
print("WARM DONE", n, len(tks), f"{time.time()-t0:.0f}s", flush=True)
