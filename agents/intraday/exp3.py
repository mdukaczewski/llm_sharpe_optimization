"""Exp 3: overnight leg conditioned on intraday shape. Strict fills: enter at close
(decide 389, fill c[390]); exit decided at bin 0, filled at c[1]. One round trip/day max."""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
from ilib import *
import warnings; warnings.filterwarnings("ignore")

def run(tk, legs_out):
    M = Min(tk); ms = M.insample & ~M.half; D = M.dates; nd = len(D)
    first30 = M.r(-1, 29); last60 = M.r(329, 389); last30 = M.r(359, 389); last120 = M.r(269, 389); mid = M.r(29, 329)
    def on_pos(side):
        pos = np.zeros((nd, 391)); pos[:, 390] = side
        pos = np.roll(pos, 0); 
        # held from c[390] of d to c[0] of d+1, then exit decided at bin 0 of d+1 fills at c[1]: hold interval 0 of d+1 too
        nxt = np.zeros((nd, 391)); nxt[1:, 0] = side[:-1]
        return pos + nxt
    out = {}
    def rep(name, side):
        g, n, t = pathbt(M, on_pos(np.nan_to_num(side)))
        d = stats(D, g, n, side, f"{tk} {name}", ms); out[name] = d; return d
    rep("ON long", np.ones(nd))
    rep("ON -sign(last30)", -np.sign(last30))
    rep("ON -sign(last60)", -np.sign(last60))
    rep("ON -sign(last120)", -np.sign(last120))
    rep("ON sign(first30)", np.sign(first30))
    z = zs(first30) - zs(last60)
    rep("ON sign(z30-z60)", np.sign(z))
    rep("ON clip(z30-z60)", np.clip(z, -2, 2) / 1.5)
    rep("ON long if last60<0", (last60 < 0) * 1.0)
    rep("ON long if z>0", (z > 0) * 1.0)
    rep("ON long if last60<0 & first30>0", ((last60 < 0) & (first30 > 0)) * 1.0)
    rep("ON long if last60<0 else short/2", np.where(last60 < 0, 1.0, -0.5))
    legs_out[tk] = out

if __name__ == "__main__":
    tks = sys.argv[1].split(",") if len(sys.argv) > 1 else ALL
    legs = {}
    for tk in tks: run(tk, legs); print()
    # cross-ticker combos
    for name in legs[tks[0]]:
        rets = [legs[t][name]["n"] for t in tks]; gs = [legs[t][name]["g"] for t in tks]
        port = combine(rets); pg = combine(gs)
        pstats(port, f"COMBO[{len(tks)}] {name}", pg)
