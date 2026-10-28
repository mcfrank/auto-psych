# name: trial_index_yes_correlation
# description: Pearson correlation between trial_index and the yes response; observed more negative than null means people say "yes" less as the session goes on (a drift the stationary model does not produce), more positive means the reverse.
def test_statistic(df):
    x = df["trial_index"].to_numpy(dtype=float)
    y = df["rating"].to_numpy(dtype=float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
