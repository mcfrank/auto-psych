# name: low_periodicity_b_choice_rate
# description: Mean chose_left when periodicity_b <= 0.667 (lowest tercile, B has low periodic structure); tests whether the model — which ignores periodicity — correctly predicts the near-ceiling left-choice rate observed for these trials.
def test_statistic(df):
    sub = df[df["periodicity_b"] <= 0.667]
    if len(sub) == 0:
        return float("nan")
    return float(sub["chose_left"].mean())
