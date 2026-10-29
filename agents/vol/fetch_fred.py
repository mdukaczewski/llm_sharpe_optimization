"""Fetch CBOE VIX (VIXCLS) and VIX3M (VXVCLS) daily closes from FRED (public CSV, no key) -> fred_vix.csv."""
import pandas as pd, urllib.request, io
out = {}
for sid, nm in [("VIXCLS", "VIX"), ("VXVCLS", "VIX3M"), ("VXNCLS", "VXN")]:
    u = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    s = urllib.request.urlopen(u, timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(s), index_col=0, parse_dates=True, na_values=["."])
    out[nm] = df.iloc[:, 0]
df = pd.DataFrame(out)
df.to_csv("/home/trumpet/strat20/agents/vol/fred_vix.csv")
print(df.dropna().tail(3)); print("rows", len(df), "first", df.index[0].date(), "VIX nan since 2014:", int(df.loc["2014":, "VIX"].isna().sum()), "VIX3M nan:", int(df.loc["2014":, "VIX3M"].isna().sum()))
