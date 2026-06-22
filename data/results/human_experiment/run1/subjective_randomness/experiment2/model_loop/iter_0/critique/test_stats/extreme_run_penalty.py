# name: extreme_run_penalty
# description: The average choice probability for the sequence with max_run_norm == 1, when compared against a sequence with max_run_norm < 1.
def test_statistic(df):
    a_extreme = df[(df["max_run_norm_a"] == 1.0) & (df["max_run_norm_b"] < 1.0)]
    b_extreme = df[(df["max_run_norm_b"] == 1.0) & (df["max_run_norm_a"] < 1.0)]

    total_extreme_choices = (
        a_extreme["chose_left"].sum() + (1 - b_extreme["chose_left"]).sum()
    )
    total_trials = len(a_extreme) + len(b_extreme)

    if total_trials == 0:
        return 0.0
    return float(total_extreme_choices / total_trials)
