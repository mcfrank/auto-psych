# name: mean_chose_left_high_periodicity_b
# description: The mean choice rate for sequence A when sequence B has high periodicity (> 0.5), which humans typically reject as non-random.
def test_statistic(df):
    subset = df[df["periodicity_b"] > 0.5]
    if len(subset) == 0:
        return 0.0
    return float(subset["chose_left"].mean())
