# name: max_run_preference
# description: The mean probability of choosing the left sequence when it has a strictly smaller maximum run length than the right sequence.
def test_statistic(df):
    mask = df['max_run_a'] < df['max_run_b']
    if not mask.any(): return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
