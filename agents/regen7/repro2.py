import sys; sys.path.insert(0,'/home/user/llm_sharpe_optimization/agents/onml4')
from lib4 import *
sys.path.insert(0, f"{HERE}/agents/onml3"); import lib3
cols=['date','tk','y_oo','tier','rv20','dv0','dvc','ldv20','half']
D=pd.read_parquet(f"{HERE}/agents/onml3/out/frame_u2000.parquet",columns=cols)
P=pd.read_parquet(f"{HERE}/agents/onml3/out/pred_f40_on.parquet")
p=D[["date","tk"]].merge(P,on=["date","tk"],how="left").p
G=D.copy(); G["p"]=p.values; G=G[G.p.notna()].reset_index(drop=True)
print(len(G), G.date.min())
G["ey"]=lib3.causal_cal(G, first_year_insample=True)
lib3.cost_cols(G,k=6,c=0.3); lib3.net_scores(G,G.ey,3.0)
r=lib3.book3(G,k=6,y="y_oo",ce="ce",cx="cx",score="sL",score_s="sS")
ref=load_ref('agents/onml3/cands/onml3_f40_on_lam3_moo_k6.csv')
print(sh(r), corr_with(r,ref)); d=(r-ref.reindex(r.index)); print(d.abs().groupby(d.index.year).mean())
H=load_G(); print('G rows', len(H), 'with y_oo', H.y_oo.notna().sum(), 'frame rows w p', len(G), G.y_oo.notna().sum())
