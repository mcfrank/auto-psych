# name: round_target_even_unmatched_gap
# description: Among even, non-example targets sharing no final digit with any example (so no example ends in 0), yes rate for round targets (multiples of 10) minus other even targets; t_observed above null_mean means people favour round numbers as program outputs beyond the model's parity credit; below means round numbers are disfavoured.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    t = df["target"].astype(int).to_numpy()
    in_set = (ex == t[:, None]).any(axis=1)
    digit_match = (np.mod(ex, 10) == (t % 10)[:, None]).any(axis=1)
    base = (~in_set) & (~digit_match) & (t % 2 == 0)
    y = df["rating"].astype(float).to_numpy()
    rnd = base & (t % 10 == 0)
    oth = base & (t % 10 != 0)
    if rnd.sum() == 0 or oth.sum() == 0:
        return 0.0
    return float(y[rnd].mean() - y[oth].mean())
