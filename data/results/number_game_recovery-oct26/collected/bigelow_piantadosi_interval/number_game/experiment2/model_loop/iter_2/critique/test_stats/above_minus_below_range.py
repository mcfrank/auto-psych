# name: above_minus_below_range
# description: For targets outside the examples' [min, max] range by 1-15, yes rate above the max minus yes rate below the min; observed above the null means people extend generalization upward (toward larger numbers) more than the model's symmetric interval hypotheses predict, below means they extend downward more.
def test_statistic(df):
    parts = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    lo = parts.min(axis=1).to_numpy()
    hi = parts.max(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    up = (t > hi) & (t - hi <= 15)
    dn = (t < lo) & (lo - t <= 15)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
