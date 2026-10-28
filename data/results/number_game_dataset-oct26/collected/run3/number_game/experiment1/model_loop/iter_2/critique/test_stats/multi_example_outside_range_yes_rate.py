# name: multi_example_outside_range_yes_rate
# description: Yes-rate for targets outside the [min, max] range of the examples on sets of 3-4 examples; observed above null means the model (whose evidence weight grows with set size) generalizes too tightly beyond large example sets, below means too loosely.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = parts.notna().sum(axis=1).to_numpy()
    lo = parts.min(axis=1).to_numpy(float)
    hi = parts.max(axis=1).to_numpy(float)
    tg = df["target"].to_numpy(float)
    ok = (n >= 3) & ((tg < lo) | (tg > hi))
    if not ok.any():
        return 0.5
    return float(df["rating"].to_numpy(float)[ok].mean())
