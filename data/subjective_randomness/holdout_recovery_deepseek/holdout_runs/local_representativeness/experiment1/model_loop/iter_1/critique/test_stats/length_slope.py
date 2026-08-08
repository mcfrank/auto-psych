# name: length_slope
# description: Linear-regression slope of chose_left on the signed length difference (n_a - n_b) in units of one symbol per unit change; probes the model's linear additive accumulation of typicality over sequence length.
def test_statistic(df):
    d = df["n_a"] - df["n_b"]
    if d.nunique() < 2:
        return 0.0
    return float(np.polyfit(d, df["chose_left"], 1)[0])
