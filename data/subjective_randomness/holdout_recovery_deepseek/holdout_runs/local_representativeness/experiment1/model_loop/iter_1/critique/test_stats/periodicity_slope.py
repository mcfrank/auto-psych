# name: periodicity_slope
# description: Linear-regression slope of chose_left on the signed difference in periodicity (periodicity_a - periodicity_b); tests whether the model reproduces preference for genuinely periodic templates, which its p+alternation mechanism ignores.
def test_statistic(df):
    d = df["periodicity_a"] - df["periodicity_b"]
    if d.nunique() < 2:
        return 0.0
    return float(np.polyfit(d, df["chose_left"], 1)[0])
