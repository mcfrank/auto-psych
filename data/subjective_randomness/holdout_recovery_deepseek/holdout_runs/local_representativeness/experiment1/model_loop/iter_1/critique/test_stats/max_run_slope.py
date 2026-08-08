# name: max_run_slope
# description: Linear-regression slope of chose_left on the signed difference in normalized max run length (max_run_norm_a - max_run_norm_b); probes whether the model captures the longer-run penalty it cannot see.
def test_statistic(df):
    d = df["max_run_norm_a"] - df["max_run_norm_b"]
    if d.nunique() < 2:
        return 0.0
    return float(np.polyfit(d, df["chose_left"], 1)[0])
