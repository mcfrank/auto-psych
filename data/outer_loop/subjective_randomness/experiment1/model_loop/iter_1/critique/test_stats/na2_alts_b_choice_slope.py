# name: na2_alts_b_choice_slope
# description: OLS regression slope of chose_left on alts_b restricted to n_a = 2 trials (the dominant trial type, 81% of data); tests whether the model correctly reproduces the steepness of the alternation-count effect on choice within the most common stimulus condition.
def test_statistic(df):
    sub = df[df["n_a"] == 2]
    if len(sub) < 3:
        return float("nan")
    x = sub["alts_b"].values.astype(float)
    y = sub["chose_left"].values.astype(float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)
