# name: above_minus_below_range_yes_rate
# description: Yes-rate for targets above the example maximum minus yes-rate for targets below the example minimum; nonzero observed vs null means people generalize asymmetrically in magnitude (positive = more upward generalization) in a way the model's symmetric intervals/rules do not reproduce.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    r = df["rating"].astype(float).values
    a, b = r[t > hi], r[t < lo]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
