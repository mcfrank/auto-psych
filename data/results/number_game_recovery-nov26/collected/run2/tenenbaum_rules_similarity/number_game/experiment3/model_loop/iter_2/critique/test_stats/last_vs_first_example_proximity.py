# name: last_vs_first_example_proximity
# description: For sets of 2+ examples, yes rate for non-example targets within 6 of the last-shown example minus yes rate for those within 6 of the first-shown example (targets near both excluded); nonzero observed vs null means example order matters (recency > 0 / primacy < 0), which the order-blind model cannot produce.
def test_statistic(df):
    parts = df["number_set"].str.split(",", expand=True).astype(float)
    n = parts.notna().sum(axis=1).values
    first = parts[0].values
    vals = parts.values
    last = vals[np.arange(len(vals)), n - 1]
    t = df["target"].astype(float).values
    isex = (vals == t[:, None]).any(axis=1)
    dl = np.abs(t - last)
    dfst = np.abs(t - first)
    base = (n >= 2) & (~isex)
    near_l = base & (dl <= 6) & (dfst > 6)
    near_f = base & (dfst <= 6) & (dl > 6)
    r = df["rating"].values.astype(float)
    if near_l.sum() == 0 or near_f.sum() == 0:
        return 0.0
    return float(r[near_l].mean() - r[near_f].mean())
