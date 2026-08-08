# name: max_run_preference
# description: Net probability of choosing the left sequence when the left has a strictly longer maximal run than when the left has a strictly shorter run, conditioning on unequal runs.
def test_statistic(df):
    a = df.loc[df['max_run_a'] > df['max_run_b']]
    b = df.loc[df['max_run_a'] < df['max_run_b']]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return a['chose_left'].mean() - b['chose_left'].mean()
