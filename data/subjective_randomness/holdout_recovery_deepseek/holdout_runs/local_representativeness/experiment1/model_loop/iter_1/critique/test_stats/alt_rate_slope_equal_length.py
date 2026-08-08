# name: alt_rate_slope_equal_length
# description: Linear-regression slope of chose_left on the signed alternation-proportion difference (p_alts_a - p_alts_b) restricted to trials where both sequences have equal length; isolates the model's core alternation mechanism at fixed evidence accumulation.
def test_statistic(df):
    sub = df[df["n_a"] == df["n_b"]]
    if len(sub) < 5:
        return 0.0
    d = sub["p_alts_a"] - sub["p_alts_b"]
    if d.nunique() < 2:
        return 0.0
    return float(np.polyfit(d, sub["chose_left"], 1)[0])
