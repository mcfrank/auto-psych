# name: fourth_example_prefix_broadening
# description: For 4-example sets whose first three examples are also shown as a 3-example set (31,3,1 -> 31,3,1,15), yes rate on the 4-example set minus the 3-example set over shared non-example targets; observed above null means the fourth example broadens generalization more (narrows less) than the model's position-decaying evidence predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    for c in range(x.shape[1], 4):
        x[c] = np.nan
    x = x.iloc[:, :4]
    X = x.to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.isfinite(X).sum(axis=1)
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    isin = (X == t[:, None]).any(axis=1)
    nd = np.nanmin(np.abs(X - t[:, None]), axis=1)
    first = X[:, 0]
    last = X[np.arange(len(X)), np.maximum(n - 1, 0)]
    def m(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    s = df["number_set"].astype(str)
    pre = s.where(pd.Series(n == 4, index=df.index)).str.rsplit(",", n=1).str[0]
    k = pd.DataFrame({"s": s.to_numpy(), "t": t, "r": r, "n": n, "pre": pre.to_numpy(), "isin": isin})
    k = k[~k["isin"]]
    three = k[k["n"] == 3].groupby(["s", "t"])["r"].mean().rename("r3").reset_index().rename(columns={"s": "pre"})
    four = k[k["n"] == 4].groupby(["pre", "t"])["r"].mean().rename("r4").reset_index()
    mm = four.merge(three, on=["pre", "t"])
    v = float((mm["r4"] - mm["r3"]).mean()) if len(mm) else 0.0
    return v if np.isfinite(v) else 0.0
