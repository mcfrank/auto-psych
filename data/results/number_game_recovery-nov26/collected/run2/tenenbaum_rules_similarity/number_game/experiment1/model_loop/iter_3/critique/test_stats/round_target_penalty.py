# name: round_target_penalty
# description: Among sets with no example divisible by 10 and targets not in the set, yes-rate for targets that are multiples of 10 minus yes-rate for other targets; observed below null means people avoid "round" targets more than the model (whose ends-in-0/multiples rules don't penalise them), above null means they favour them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    has10 = np.nansum((ex % 10 == 0) & ~np.isnan(ex), axis=1) > 0
    member = np.nansum(ex == t[:, None], axis=1) > 0
    keep = ~has10 & ~member
    rnd = keep & (t % 10 == 0)
    oth = keep & (t % 10 != 0)
    if rnd.sum() == 0 or oth.sum() == 0:
        return 0.0
    return float(r[rnd].mean() - r[oth].mean())
