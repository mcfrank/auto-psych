# name: alts_b5_choice_rate
# description: Mean chose_left when alts_b = 5 (moderately-high alternation in B, the critical transition zone between majority A-choice at alts_b = 4 and full B-choice at alts_b = 7); tests model calibration at the steepest part of the alternation-rate psychometric function.
def test_statistic(df):
    sub = df[df["alts_b"] == 5]
    if len(sub) == 0:
        return float("nan")
    return float(sub["chose_left"].mean())
