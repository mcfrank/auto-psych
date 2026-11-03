# name: mixed_four_example_range_gap
# description: In 4-example sets mixing odd and even numbers, yes rate for non-example targets inside [min,max] of the examples minus targets outside it; observed above null means people generalize by numerical range in rule-less sets more sharply than the model predicts.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pad = np.isnan(E)
    n = (~pad).sum(axis=1)
    in_set = (E == t[:, None]).any(axis=1)
    def rate(mask):
        return float(y[mask].mean()) if mask.any() else 0.0
    odd = ((E % 2 == 1) & ~pad).any(axis=1)
    even = ((E % 2 == 0) & ~pad).any(axis=1)
    lo, hi = np.nanmin(E, axis=1), np.nanmax(E, axis=1)
    base = (n == 4) & odd & even & ~in_set
    inside = (t >= lo) & (t <= hi)
    return rate(base & inside) - rate(base & ~inside)
