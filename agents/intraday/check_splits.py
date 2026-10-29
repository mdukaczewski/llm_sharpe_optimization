"""Verify minute-derived close-to-close returns (with my split adjustment) vs panel_liquid.npz."""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday"); sys.path.insert(0, "/home/trumpet/strat20")
from ilib import *
from bt import Panel
P = Panel(path="/home/trumpet/strat20/data/panel_liquid.npz")
for tk in ["SPY","QQQ","AAPL","MSFT","NVDA","AMZN","TSLA","IWM","XLK"]:
    M = Min(tk); r = path_returns(M)
    cc_min = pd.Series(M.cc, index=M.dates)
    j = P.col(tk); pc = pd.Series(P.close[:, j], index=P.dates); pcc = pc / pc.shift(1) - 1
    d = (cc_min - pcc.reindex(M.dates)).dropna()
    big = d[d.abs() > 0.004]
    on = pd.Series(r[:, 390], index=M.dates)
    print(f"{tk}: n={len(d)} median|diff|={d.abs().median()*1e4:.2f}bps  n(|diff|>40bps)={len(big)}  worst: {big.abs().sort_values().tail(3).to_dict()}  min ON ret {on.min()*100:.1f}% on {on.idxmin().date()}  max {on.max()*100:.1f}%")
