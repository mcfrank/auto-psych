# name: round_target_bonus_no_round_examples
# description: Among sets containing no multiple of 10, yes rate for targets that are multiples of 10 minus yes rate for other targets; observed above null means people say yes to round numbers more than the model predicts when nothing in the examples supports them (a round-number salience), below null means less.
def test_statistic(df):
    s = df["number_set"].astype(str)
    parts = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    has_round = np.nansum((np.mod(parts, 10) == 0).astype(float), axis=1) > 0
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    keep = ~has_round
    rt = keep & (t % 10 == 0)
    ot = keep & (t % 10 != 0)
    if rt.sum() == 0 or ot.sum() == 0:
        return 0.0
    return float(r[rt].mean() - r[ot].mean())
