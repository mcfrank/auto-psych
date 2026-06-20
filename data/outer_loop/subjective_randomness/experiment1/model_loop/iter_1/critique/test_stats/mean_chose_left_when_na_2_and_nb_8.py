# name: mean_chose_left_when_na_2_and_nb_8
# description: The overall choice rate for sequence A when comparing a length-2 sequence to a length-8 sequence.
def test_statistic(df):
    subset = df[(df["n_a"] == 2) & (df["n_b"] == 8)]
    if len(subset) == 0:
        return 0.0
    return float(subset["chose_left"].mean())
