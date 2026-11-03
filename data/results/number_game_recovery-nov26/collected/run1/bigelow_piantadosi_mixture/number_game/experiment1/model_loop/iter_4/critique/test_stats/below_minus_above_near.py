# name: below_minus_above_near
# description: Yes-rate for targets 1-10 below the set minimum minus yes-rate for targets 1-10 above the set maximum (multi-example sets); observed < null means people extend upward relatively more (downward less) than the model predicts, observed > null the reverse.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    n = parts.notna().sum(axis=1).values
    lo = parts.min(axis=1).values
    hi = parts.max(axis=1).values
    t = df["target"].astype(float).values
    y = df["rating"].values.astype(float)
    below = (n >= 2) & (lo - t >= 1) & (lo - t <= 10)
    above = (n >= 2) & (t - hi >= 1) & (t - hi <= 10)
    if below.sum() == 0 or above.sum() == 0:
        return 0.0
    return float(y[below].mean() - y[above].mean())
