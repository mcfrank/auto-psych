# name: periodicity_penalty
# description: Net probability of choosing the left sequence when the left has higher periodicity than when the left has lower periodicity, conditioning on unequal periodicity measures.
def test_statistic(df):
    a = df.loc[df['periodicity_a'] > df['periodicity_b']]
    b = df.loc[df['periodicity_a'] < df['periodicity_b']]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return a['chose_left'].mean() - b['chose_left'].mean()
