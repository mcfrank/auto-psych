# name: target_one_yes
# description: Yes rate when the target is 1 and 1 is not among the examples; observed above null means people accept 1 more often than the model's rules (odd, squares, powers, ends/contains 1) and ranges imply, below means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    has1 = (ex == 1).any(axis=1)
    m = (df["target"].astype(int).to_numpy() == 1) & ~has1
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy(dtype=float)[m].mean())
