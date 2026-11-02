# name: split_cell_fraction
# description: Fraction of (set, target) cells with at least 5 answers whose answers are not unanimous; observed above null means people disagree on the same item more than the model's per-item yes probabilities imply (they are too sharp), below means people agree more than predicted.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    nd = x.sub(t, axis=0).abs().min(axis=1)
    key = df["number_set"].astype(str) + "|" + t.astype(str)
    g = r.groupby(key).agg(["size", "mean"])
    g = g[g["size"] >= 5]
    v = ((g["mean"] > 0) & (g["mean"] < 1)).mean()
    return float(0.0 if (v is None or not np.isfinite(v)) else v)
