# name: parity_match_effect
# description: Among sets of 2+ examples that all share a parity, yes rate for same-parity targets minus yes rate for opposite-parity targets; observed above null_mean means people weight parity more than the model, below means the model overweights the even/odd rules.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1)
    odd = (s % 2).sum(axis=1)
    all_odd = odd == n
    all_even = odd == 0
    m = (n >= 2) & (all_odd | all_even)
    t_odd = df["target"].astype(int) % 2 == 1
    same = m & (t_odd == all_odd)
    diff = m & (t_odd != all_odd)
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(df.loc[same, "rating"].mean() - df.loc[diff, "rating"].mean())
