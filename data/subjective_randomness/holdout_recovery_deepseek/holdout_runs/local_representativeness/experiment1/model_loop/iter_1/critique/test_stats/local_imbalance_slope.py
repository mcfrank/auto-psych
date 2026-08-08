# name: local_imbalance_slope
# description: Linear-regression slope of chose_left on the signed difference in worst local (length-4) imbalance (local_imbalance_a - local_imbalance_b); probes positional imbalancers the global-p model cannot represent.
def test_statistic(df):
    d = df["local_imbalance_a"] - df["local_imbalance_b"]
    if d.nunique() < 2:
        return 0.0
    return float(np.polyfit(d, df["chose_left"], 1)[0])
