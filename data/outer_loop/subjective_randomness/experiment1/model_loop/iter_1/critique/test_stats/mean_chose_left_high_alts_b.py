# name: mean_chose_left_high_alts_b
# description: The choice rate for sequence A when sequence B has many alternations (>= 5), probing the over-alternation penalty.
def test_statistic(df):
    subset = df[df["alts_b"] >= 5]
    if len(subset) == 0:
        return 0.0
    return float(subset["chose_left"].mean())
