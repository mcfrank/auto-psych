# name: local_imbalance_sensitivity
# description: Net probability of choosing the left sequence when the left has larger within-window local imbalance than when the left has smaller local imbalance, conditioning on unequal values.
def test_statistic(df):
    a = df.loc[df['local_imbalance_a'] > df['local_imbalance_b']]
    b = df.loc[df['local_imbalance_a'] < df['local_imbalance_b']]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return a['chose_left'].mean() - b['chose_left'].mean()
