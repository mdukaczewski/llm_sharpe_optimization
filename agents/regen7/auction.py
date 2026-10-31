import numpy as np, pandas as pd, os, sys
os.environ["STRAT_END"]="2026-10-28"; sys.path.insert(0,"/home/user/llm_sharpe_optimization"); import rlib
from rlib import Min
Z=np.load("/home/user/llm_sharpe_optimization/data/panel_liquid.npz",allow_pickle=True); dates=pd.DatetimeIndex(Z["dates"]); ti={t:i for i,t in enumerate(Z["tickers"])}
T=pd.read_csv("/home/user/llm_sharpe_optimization/out/quotes_trades.csv",parse_dates=["date","d_out"]); T=T[T.book=="onml_k10"]
rows=[]
for tk,g in list(T.groupby("tk"))[::6]:
    try: M=Min(tk,end="2026-10-28")
    except Exception: continue
    di={d:i for i,d in enumerate(M.dates)}
    for _,r in g.iterrows():
        i=di.get(r.d_out); j=di.get(r.date)
        if i is None or j is None: continue
        o_off=Z["open"][dates.get_loc(r.d_out),ti[tk]] if r.d_out in dates and tk in ti else np.nan
        rows.append((tk,r["rank"],M.v[i,0]*M.vw[i,0], M.v[j,390]*M.vw[j,390], M.o_raw[i,0], o_off, M.c[i,1]))
D=pd.DataFrame(rows,columns=["tk","rank","dv_bin0","dv_bin390","o_min","o_off","c0931"])
D["tier"]=pd.cut(D["rank"],[0,300,600,1500,2000,9e9],labels=["1-300","301-600","601-1500","1501-2000","x"])
D["part_open"]=2e4/D.dv_bin0; D["part_close"]=2e4/D.dv_bin390; D["dopen_bps"]=(D.o_min/D.o_off-1).abs()*1e4
pd.set_option("display.width",200)
print(D.groupby("tier",observed=True).agg(n=("tk","size"),dv0_med=("dv_bin0","median"),dv390_med=("dv_bin390","median"),part_open_med=("part_open","median"),part_open_p75=("part_open",lambda x:x.quantile(.75)),part_close_med=("part_close","median"),open_mismatch_med_bps=("dopen_bps","median"),open_mismatch_p90=("dopen_bps",lambda x:x.quantile(.9))).round(3).to_string())
