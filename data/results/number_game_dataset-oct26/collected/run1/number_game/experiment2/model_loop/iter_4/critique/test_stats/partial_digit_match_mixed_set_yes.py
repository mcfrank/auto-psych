# name: partial_digit_match_mixed_set_yes
# description: In sets of >=3 examples with >=2 distinct final digits, yes rate for non-example targets whose final digit matches exactly one example; t_observed above null_mean means a single digit match counts as a full match (max-like), not the 1/n share the model's averaged similarity gives; below means partial matches count even less than averaged.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    n = np.isfinite(ex).sum(axis=1)
    d = np.mod(ex, 10)
    nd = pd.DataFrame(d).nunique(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    in_set = (ex == t[:, None]).any(axis=1)
    k = (d == (t % 10)[:, None]).sum(axis=1)
    m = (n >= 3) & (nd >= 2) & (~in_set) & (k == 1)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).to_numpy()[m].mean())
