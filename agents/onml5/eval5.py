#!/usr/bin/env python3
"""Evaluate prediction sets through the UNCHANGED onml4 pipeline (lib4.prep: causal calibration + decay ratios; route_book k6
lam 3, entry route MOC / 15:59:30 cross, exit MOO or routed; c = 0.3, f = 1; fold-OOS at the $0.4M commission; c = 0.1 / 1.0
with the selection re-run at that c).  Also per-day rank IC vs y_oo within each tier.
usage: python3 eval5.py <spec> [<spec> ...] [tag=...] [sens=0|1]
  spec = name           -> agents/onml5/out/pred_<name>.parquet
       = o3:name        -> agents/onml3/out/pred_<name>.parquet
       = A+B            -> per-day rank average of two specs (each ranked first)
       = splice:A:B     -> A with its ordering inside the dynamic top-600 replaced by B's (quantile swap, see splice())
  built-in: p40, p26, pR (the saved quarterly 3-seed onml3 predictions and their onml4 rank average)."""
from lib5 import *
args = [a for a in sys.argv[1:] if "=" not in a]; kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
TAG = kw.get("tag", "eval"); SENS = kw.get("sens", "1") == "1"
G = load_G(); G["pR"] = prank((G.p40 + G.p26) / 2, G.date)
X5 = pd.read_parquet(f"{OUT}/feat5.parquet", columns=["date", "tk", "drank"]); G = G.merge(X5, on=["date", "tk"], how="left"); del X5
cache = {}
def load(spec):
    if spec in G.columns: return G[spec]
    if spec in cache: return cache[spec]
    if spec.startswith("splice:"):
        _, a, b = spec.split(":"); s = splice(load(a), load(b))
    elif "+" in spec:
        parts = spec.split("+"); s = prank(sum(prank(load(x), G.date) for x in parts) / len(parts), G.date)
    else:
        path = f"{HERE}/agents/onml3/out/pred_{spec[3:]}.parquet" if spec.startswith("o3:") else f"{OUT}/pred_{spec}.parquet"
        P = pd.read_parquet(path, columns=["date", "tk", "p"]); s = G[["date", "tk"]].merge(P, on=["date", "tk"], how="left").p
        s.index = G.index; s = prank(s, G.date)
    cache[spec] = s; return s

def splice(a, b):
    """inside each day's dynamic top-600 (drank <= 600) reassign a's values in b's order (a's group distribution kept, so the
    global calibration map is unchanged except for the ordering inside the liquid group); rows beyond keep a."""
    out = a.copy(); m = (G.drank <= 600) & a.notna() & b.notna()
    sub = pd.DataFrame({"date": G.date[m], "a": a[m], "b": b[m]})
    sub["ra"] = sub.groupby("date").a.rank(method="first"); sub["rb"] = sub.groupby("date").b.rank(method="first")
    srt = sub.sort_values(["date", "ra"]); lut = dict(zip(zip(srt.date, srt.ra), srt.a))
    out[m] = [lut[(d, r)] for d, r in zip(sub.date, sub.rb)]
    return prank(out, G.date)

def ic_row(spec, s):
    g = G[["date", "tier", "y_oo"]].assign(p=s.values); row = dict(pred=spec)
    v = daily_ic(g, "p", "y_oo"); row["ic_all"] = round(v.mean(), 4)
    v = daily_ic(g, "p", "y_oo", by="tier").reset_index(); v.columns = ["date", "tier", "ic"]
    for t, x in v.groupby("tier"): row["ic_" + TNAMES[t]] = round(x.ic.mean(), 4)
    x = v[v.tier == 0]; row["ic1_h1"] = round(x.ic[x.date < "2021-10-28"].mean(), 4); row["ic1_h2"] = round(x.ic[x.date >= "2021-10-28"].mean(), 4)
    return row
REF = load_ref(REF_RTX); rows = []; ics = []
for spec in args:
    s = load(spec); G["_p"] = s.values; ics.append(ic_row(spec, s))
    rr, ser, tabs = books_for(G, "_p", ks=(6,), sens=SENS, tiers=True)
    for row in rr:
        row["book"] = row["book"].replace("_p_", spec + " "); nm = [k for k in ser if k.endswith(row["book"].split(" ")[-1])][0]
        row["corr_rtx"] = round(corr_with(commission(ser[nm], 6), commission(REF, 6)), 2); rows.append(row)
        log(f"{TAG} " + str({k: v for k, v in row.items()}))
    for t in tabs: t["book"] = t.book.str.replace("_p_", spec + " "); log(f"{TAG} tiers\n" + t.to_string(index=False))
    for nm, r in ser.items(): r.rename(f"{spec} {nm[3:]}").to_frame().to_parquet(f"{OUT}/ser_{TAG}_{spec.replace(':', '_').replace('+', '_')}_{nm[3:]}.parquet")
T = pd.DataFrame(rows); I = pd.DataFrame(ics)
tlog(f"{TAG} IC vs y_oo (per-day Spearman; tier columns = within tier; ic1_h1/h2 = ranks 1-300 halves)", I)
tlog(f"{TAG} books k6 lam3 c=0.3 f=1 ($0.4M)", T.drop(columns=["yrs"]))
tlog(f"{TAG} by year", T[["book", "yrs"]])
T.to_csv(f"{OUT}/eval_{TAG}.csv", index=False); I.to_csv(f"{OUT}/ic_{TAG}.csv", index=False)
log(f"{TAG} DONE")
