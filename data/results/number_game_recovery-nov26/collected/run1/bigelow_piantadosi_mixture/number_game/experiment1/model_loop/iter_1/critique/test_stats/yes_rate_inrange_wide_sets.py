# name: yes_rate_inrange_wide_sets
# description: Yes rate for non-example targets strictly inside the [min, max] range of multi-example sets spanning at least 40; observed below the null means the model over-endorses the interior of wide example ranges (humans do not fill in broad intervals).
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = (~np.isnan(sets)).sum(axis=1)
    lo, hi = np.nanmin(sets, axis=1), np.nanmax(sets, axis=1)
    t = df["target"].astype(float).values
    member = (sets == t[:, None]).any(axis=1)
    mask = (n >= 2) & (hi - lo >= 40) & (t > lo) & (t < hi) & ~member
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
