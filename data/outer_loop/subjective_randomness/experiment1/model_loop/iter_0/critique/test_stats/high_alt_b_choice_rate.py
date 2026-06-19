# name: high_alt_b_choice_rate
# description: Mean chose_left when p_alts_b > 0.65 (B is highly alternating); tests whether the model correctly predicts choice direction when B switches flanks at a high rate.
def test_statistic(df):
    mask = df['p_alts_b'] > 0.65
    sub = df[mask]
    if len(sub) == 0:
        return float('nan')
    return float(sub['chose_left'].mean())
