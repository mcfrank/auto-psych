# name: low_alt_b_choice_rate
# description: Mean chose_left when p_alts_b < 0.45 (B is streaky, few alternations); tests whether the model correctly predicts choice when B contains long same-symbol runs.
def test_statistic(df):
    mask = df['p_alts_b'] < 0.45
    sub = df[mask]
    if len(sub) == 0:
        return float('nan')
    return float(sub['chose_left'].mean())
