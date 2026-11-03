# name: cell_mixed_response_fraction
# description: Fraction of distinct (number_set, target) cells whose repeated answers are not unanimous (some yes, some no); t_observed below null_mean means people agree more (more all-or-none) than the model's graded per-cell probabilities predict, above means they disagree more.
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(int).astype(str)
    g = df["rating"].astype(float).groupby(key.to_numpy())
    s = g.sum(); n = g.size()
    mixed = (s > 0) & (s < n) & (n > 1)
    denom = (n > 1).sum()
    return float(mixed.sum() / denom) if denom > 0 else 0.0
