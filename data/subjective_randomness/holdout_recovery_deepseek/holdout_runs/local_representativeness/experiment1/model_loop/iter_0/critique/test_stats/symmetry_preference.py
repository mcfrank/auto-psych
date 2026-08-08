# name: symmetry_preference
# description: Net probability of choosing the left sequence when the left has more symmetric-structure motifs (sum of sym1-sym8) than when the left has fewer, conditioning on unequal symmetric motif counts.
def test_statistic(df):
    sym_a = sum(df[f'sym{i}_a'] for i in range(1, 9))
    sym_b = sum(df[f'sym{i}_b'] for i in range(1, 9))
    a = df.loc[sym_a > sym_b]
    b = df.loc[sym_a < sym_b]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return a['chose_left'].mean() - b['chose_left'].mean()
