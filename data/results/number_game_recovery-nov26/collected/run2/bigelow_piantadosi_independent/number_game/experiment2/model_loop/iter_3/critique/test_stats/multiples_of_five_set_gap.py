# name: multiples_of_five_set_gap
# description: In sets of >=2 examples that are all multiples of 5, yes rate for non-example multiples of 5 minus yes rate for non-multiples; observed below null means people generalize such sets to all multiples of 5 less than the split-out multiples-of-5 prior predicts, above means more.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pad = np.isnan(E)
    n = (~pad).sum(axis=1)
    in_set = (E == t[:, None]).any(axis=1)
    def rate(mask):
        return float(y[mask].mean()) if mask.any() else 0.0
    all_five = ((E % 5 == 0) | pad).all(axis=1) & (n >= 2)
    base = all_five & ~in_set
    return rate(base & (t % 5 == 0)) - rate(base & (t % 5 != 0))
