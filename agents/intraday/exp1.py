"""Exp 1: broad screen of classic intraday patterns, all tickers, 2016-10-28..2026-10-27.
Legs are evaluated with fill-at-next-bar and per-side costs."""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
from ilib import *
import warnings; warnings.filterwarnings("ignore")

def rvol(M, w=20):
    return pd.Series(M.cc).rolling(w).std().shift(1).values

def screen(tk):
    M = Min(tk); ms = M.insample & ~M.half
    D = M.dates
    print(f"\n===== {tk}  days={ms.sum()} cost={M.cost}bps/side  check: mean|c390/c389-1| = {np.nanmean(np.abs(M.c[:,390]/M.c[:,389]-1))*1e4:.1f}bps")
    res = {}
    # 1. first-30-min momentum -> last 30 min (Gao et al.): signal r(open..bin29), decide at 359, in at 360, out at close
    r30 = M.r(-1, 29)
    side = np.sign(r30)
    g, n = leg(M, side, 359, 389); res["mom30_last30"] = stats(D, g, n, side, f"{tk} first30->last30 sign", ms)
    # 1b. sign of 12:30-13:00 (bins 180..209) -> last 30
    rmid = M.r(180, 209); side = np.sign(rmid)
    g, n = leg(M, side, 359, 389); stats(D, g, n, side, f"{tk} 12:30-13:00 -> last30", ms)
    # 1c. combined open-to-15:30 return sign -> last 30
    rd = M.r(-1, 359); side = np.sign(rd)
    g, n = leg(M, side, 359, 389); res["day_last30"] = stats(D, g, n, side, f"{tk} open->15:30 -> last30", ms)
    # 1d. open->15:00 -> last hour
    rd2 = M.r(-1, 329); side = np.sign(rd2)
    g, n = leg(M, side, 329, 389); stats(D, g, n, side, f"{tk} open->15:00 -> last60", ms)
    # 2. overnight gap -> first 30 min continuation (+1) / reversal (-1)
    side = np.sign(M.gap)
    g, n = leg(M, side, 0, 29); res["gap_first30"] = stats(D, g, n, side, f"{tk} gap sign -> 09:31-10:00", ms)
    g, n = leg(M, side, 0, 389); stats(D, g, n, side, f"{tk} gap sign -> 09:31-close", ms)
    g, n = leg(M, -side, 0, 389); stats(D, g, n, -side, f"{tk} gap REVERSAL 09:31-close", ms)
    # 2b large gaps only (> 1 sd of 20d cc vol)
    v = rvol(M); big = np.abs(M.gap) > v
    side = np.where(big, np.sign(M.gap), 0)
    g, n = leg(M, side, 0, 389); stats(D, g, n, side, f"{tk} big gap cont 09:31-close", ms)
    # 3. ORB: 30-min range breakout, trade until close, decide at first bin after 10:00 where c > high or < low
    hi = np.nanmax(M.h_raw[:, :30], 1); lo = np.nanmin(M.l_raw[:, :30], 1)
    c = M.c
    up = c[:, 30:] > hi[:, None]; dn = c[:, 30:] < lo[:, None]
    first_up = np.where(up.any(1), up.argmax(1) + 30, 999); first_dn = np.where(dn.any(1), dn.argmax(1) + 30, 999)
    m_in = np.minimum(first_up, first_dn); side = np.where(first_up < first_dn, 1.0, np.where(first_dn < first_up, -1.0, 0.0))
    side[m_in >= 370] = 0
    pin = np.array([c[i, min(m + 1, 390)] if m < 999 else np.nan for i, m in enumerate(m_in)])
    g = side * (c[:, 390] / pin - 1); g = np.nan_to_num(g); n = g - np.abs(side) * 2 * M.cost / 1e4
    res["orb30"] = stats(D, g, n, side, f"{tk} ORB30 -> close", ms)
    # 4. time-of-day seasonality: unconditional long each 30-min block (no cost since held through? apply cost)
    for a, b, lab in [(0, 29, "09:31-10:00"), (29, 59, "10:00-10:30"), (59, 119, "10:30-11:30"), (119, 209, "11:30-13:00"), (209, 299, "13:00-14:30"), (299, 359, "14:30-15:30"), (359, 389, "15:30-16:00")]:
        g, n = leg(M, np.ones(len(D)), a, b); stats(D, g, n, None, f"{tk} long {lab}", ms)
    # 5. VWAP reversion: at 15:00 (bin 329) z = (c - cvwap)/ (intraday vol); short if above, long if below, hold to close
    dev = (c[:, 329] / M.cvwap[:, 329] - 1)
    side = -np.sign(dev)
    g, n = leg(M, side, 329, 389); res["vwap_rev"] = stats(D, g, n, side, f"{tk} VWAP reversion 15:00->close", ms)
    g, n = leg(M, -side, 329, 389); stats(D, g, n, side, f"{tk} VWAP momentum 15:00->close", ms)
    # 6. overnight leg only (buy close, sell next open) with next-open fill = c[0] (first bar close) : cost
    side = np.ones(len(D)); g = M.c[:, 0] / M.prev_close - 1; g = np.nan_to_num(g); n = g - 2 * M.cost / 1e4
    stats(D, g, n, side, f"{tk} overnight close->09:31", ms)
    # 7. first-30 momentum sized by 1/vol and conditioned on big first-30 move
    v30 = pd.Series(np.abs(r30)).rolling(20).mean().shift(1).values
    side = np.where(np.abs(r30) > v30, np.sign(r30), 0)
    g, n = leg(M, side, 359, 389); stats(D, g, n, side, f"{tk} first30 big-only -> last30", ms)
    return res

if __name__ == "__main__":
    tks = sys.argv[1].split(",") if len(sys.argv) > 1 else ALL
    for tk in tks:
        try: screen(tk)
        except Exception as e: print(tk, "ERR", e)
