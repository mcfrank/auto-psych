# name: occ_n50_sensitivity
# description: Net probability of choosing the left sequence when the left has a higher overlapping-occurrence statistic over a length-50 window than the right, conditioning on unequal window-50 occurrence values.
def test_statistic(df):
    a = df.loc[df['occ_n50_a'] > df['occ_n50_b']]
    b = df.loc[df['occ_n50_a'] < df['occ_n50_b']]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return a['chose_left'].mean() - b['chose_left'].mean()
