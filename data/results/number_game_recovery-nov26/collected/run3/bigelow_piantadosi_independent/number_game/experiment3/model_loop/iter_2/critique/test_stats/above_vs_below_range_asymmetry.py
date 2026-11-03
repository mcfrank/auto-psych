# name: above_vs_below_range_asymmetry
# description: Among sets with 2+ examples, yes rate for targets above the largest example minus yes rate for targets below the smallest example; observed above null means people extend generalisation upward more than downward relative to the model's symmetric ranges and rules, below null means more downward.
def test_statistic(df):
    s = df["number_set"].astype(str)
    parts = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    mx = parts.max(axis=1).to_numpy()
    mn = parts.min(axis=1).to_numpy()
    multi = (s.str.count(",") >= 1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    up = multi & (t > mx)
    dn = multi & (t < mn)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
