# name: head_proportion_slope
# description: Linear-regression slope of chose_left on the signed head-proportion difference (p_a - p_b); probes the model's symmetric-prototype imbalance penalty for sign and non-linearity in regime (over/under-representation of heads).
def test_statistic(df):
    d = df["p_a"] - df["p_b"]
    if d.nunique() < 2:
        return 0.0
    return float(np.polyfit(d, df["chose_left"], 1)[0])
