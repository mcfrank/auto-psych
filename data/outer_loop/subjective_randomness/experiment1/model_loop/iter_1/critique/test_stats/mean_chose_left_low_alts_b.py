# name: mean_chose_left_low_alts_b
# description: The choice rate for sequence A when sequence B has very few alternations (<= 2).
def test_statistic(df):
    subset = df[df["alts_b"] <= 2]
    if len(subset) == 0:
        return 0.0
    return float(subset["chose_left"].mean())
