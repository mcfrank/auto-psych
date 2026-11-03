# name: prefix_extension_yes_change
# description: For 4-example sets whose first three examples are also shown as a 3-example set, yes rate on the 4-example set minus the 3-example set over the same non-example targets; observed above null means adding a fourth example narrows generalization less (or broadens it more) than the model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    nd = x.sub(t, axis=0).abs().min(axis=1)
    s = df["number_set"].astype(str)
    pre = s.where(n == 4).str.rsplit(",", n=1).str[0]
    k = pd.DataFrame({"s": s, "t": t, "r": r, "n": n, "pre": pre, "isin": isin})
    k = k[~k["isin"]]
    three = k[k["n"] == 3].groupby(["s", "t"])["r"].mean().rename("r3").reset_index().rename(columns={"s": "pre"})
    four = k[k["n"] == 4].groupby(["pre", "t"])["r"].mean().rename("r4").reset_index()
    m = four.merge(three, on=["pre", "t"])
    v = (m["r4"] - m["r3"]).mean() if len(m) else 0.0
    return float(0.0 if (v is None or not np.isfinite(v)) else v)
