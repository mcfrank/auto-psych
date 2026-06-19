# name: high_max_run_b_choice_rate
# description: Mean chose_left when max_run_norm_b > 0.4 (B has a long unbroken run relative to its length); tests if the model captures run-length salience as a non-randomness cue.
def test_statistic(df):
    mask = df['max_run_norm_b'] > 0.4
    sub = df[mask]
    if len(sub) == 0:
        return float('nan')
    return float(sub['chose_left'].mean())
