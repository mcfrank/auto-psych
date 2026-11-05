# name: odd_target_after_all_even_set_rate
# description: Yes-rate for odd targets when every example in the set is even; observed > null means the model under-produces cross-parity (similarity/noise) generalization, observed < null means it over-produces it.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    u = pd.Series(sets.unique())
    alleven = {s: all(int(x) % 2 == 0 for x in s.split(",")) for s in u}
    m = sets.map(alleven).values & (df["target"].astype(int).values % 2 == 1)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
