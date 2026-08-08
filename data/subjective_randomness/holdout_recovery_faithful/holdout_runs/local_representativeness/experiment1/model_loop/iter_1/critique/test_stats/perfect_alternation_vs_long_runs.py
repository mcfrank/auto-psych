# name: perfect_alternation_vs_long_runs
# description: The mean probability of choosing the left sequence when it is perfectly alternating (p_alts=1) and the right sequence has long runs (max_run >= 4).
def test_statistic(df):
    mask = (df['p_alts_a'] == 1.0) & (df['max_run_b'] >= 4)
    if not mask.any(): return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
