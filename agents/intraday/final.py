"""FINAL intraday strategy (per ticker, one netted position path per day):
  ON  : at the close (decision bin 389 -> MOC at official close), long overnight with weight
        w = clip(a*z30 - z60, 0, cap), z30 = first-30-min return / its trailing 20d sd,
        z60 = last-60-min return / its trailing 20d sd. Exit decided at 09:30 (bin 0), filled c[09:31].
  AM  : 09:31-10:00 reversal of yesterday's first-30-min return when |z| > dz (fill c[09:31], exit c[10:00])
  PM  : 15:00-15:30 continuation of open->15:00 when |z| > dz (fill c[15:01], exit c[15:31])
  Costs on every position change (netted inside the day). Half days: flat. Cross-ticker: equal
  trailing-60d-vol weights, 10%/yr target. Reports gross/net Sharpe etc. per ticker and combined.
usage: final.py TICKERS [a=0.5] [cap=1.5] [dz=1.0] [legs=ON,AM,PM] [dec=389] [tag]"""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
from ilib import *
import warnings; warnings.filterwarnings("ignore")
LAG = lambda x: np.r_[np.nan, x[:-1]]
def zsd(x, w=20):
    s = pd.Series(x); return (s / s.rolling(w, min_periods=10).std().shift(1)).values

def build(M, a=0.5, cap=1.5, dz=1.0, legs=("ON", "AM", "PM"), dec=389, zw=20):
    nd = len(M.dates); pos = np.zeros((nd, 391))
    if "ON" in legs:
        z60 = np.nan_to_num(zsd(M.r(dec - 60, dec), zw)); z30 = np.nan_to_num(zsd(M.r(-1, 29), zw))
        w = np.clip(a * z30 - z60, 0, cap)
        w[M.half] = 0
        if dec == 389: pos[:, 390] += w
        else: pos[:, dec + 1:391] += w[:, None]
        pos[1:, 0] += w[:-1]
    if "PM" in legs:
        z = np.nan_to_num(zsd(M.r(-1, 329), zw)); s = np.sign(z) * (np.abs(z) > dz); s[M.half] = 0
        pos[:, 330:360] += s[:, None]
    if "AM" in legs:
        z = np.nan_to_num(zsd(LAG(M.r(-1, 29)), zw)); s = -np.sign(z) * (np.abs(z) > dz); s[M.half] = 0
        pos[:, 1:30] += s[:, None]
    return pos

def run_ticker(tk, **kw):
    M = Min(tk); ms = M.insample
    pos = build(M, **kw)
    g, n, t = pathbt(M, pos)
    d = stats(M.dates, g, n, None, tk, ms, p=False); d["trades"] = t[ms].sum() / (ms.sum() / 252)
    d["expo"] = np.abs(pos[ms]).mean() * 1.0
    return d

def line(name, d):
    print(f"{name:26s} Sh gross {d['sh_g']:5.2f} net {d['sh_n']:5.2f} | ret {d['ret']*100:6.2f}% vol {d['vol']*100:5.2f}% mdd {d['mdd']*100:6.1f}% | trades/yr {d['trades']:5.0f} | yrs " + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in d["yr"].items()), flush=True)

def combo(ds, name):
    tks = list(ds)
    port = combine([ds[t]["n"] for t in tks]); pg = combine([ds[t]["g"] for t in tks])
    eq = (1 + port).cumprod(); dd = (eq / eq.cummax() - 1).min()
    yr = port.groupby(port.index.year).apply(sh)
    h1 = port[port.index < "2021-10-28"]; h2 = port[port.index >= "2021-10-28"]
    print(f"{name:26s} Sh gross {sh(pg):5.2f} net {sh(port):5.2f} | ret {port.mean()*252*100:6.2f}% vol {port.std()*np.sqrt(252)*100:5.2f}% mdd {dd*100:6.1f}% | halves {sh(h1):.2f}/{sh(h2):.2f} | yrs " + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()), flush=True)
    return port, pg

if __name__ == "__main__":
    tks = sys.argv[1].split(",")
    kw = dict(a=0.5, cap=1.5, dz=1.0, dec=389, legs=("ON", "AM", "PM")); tag = "final"
    for arg in sys.argv[2:]:
        k, v = arg.split("=")
        if k == "legs": kw[k] = tuple(v.split(","))
        elif k == "tag": tag = v
        elif k == "dec": kw[k] = int(v)
        else: kw[k] = float(v)
    print("config:", kw, "tickers:", tks)
    ds = {}
    for tk in tks:
        ds[tk] = run_ticker(tk, **kw); line(tk, ds[tk])
    port, pg = combo(ds, f"COMBO[{len(tks)}]")
    L = "/home/trumpet/strat20/agents/intraday/logs"
    pd.DataFrame({"ret": port}).rename_axis("date").to_csv(f"{L}/{tag}_combo_net.csv", float_format="%.8f")
    pd.DataFrame({"ret": pg}).rename_axis("date").to_csv(f"{L}/{tag}_combo_gross.csv", float_format="%.8f")
    for tk in tks:   # ticker-level book: daily net return per 1x notional (unlevered)
        ds[tk]["n"].rename("ret").rename_axis("date").to_csv(f"{L}/{tag}_{tk}_net.csv", float_format="%.8f")
    # leg attribution (each leg alone, same universe)
    for legs in [("ON",), ("AM",), ("PM",), ("ON", "AM"), ("ON", "PM")]:
        d2 = {tk: run_ticker(tk, **{**kw, "legs": legs}) for tk in tks}
        combo(d2, f"COMBO legs={'+'.join(legs)}")
