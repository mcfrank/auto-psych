# name: shared_last_digit_match_effect
# description: Among multi-example sets whose examples all end in the same digit, yes-rate on targets (not in the set) ending in that digit minus targets ending in a different digit; observed above null means the model under-weights "ends in d" rules when the evidence clearly supports them, below null means it over-weights them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    ld = np.where(valid, np.mod(ex, 10), np.nan)
    first = ld[:, 0]
    shared = (np.where(valid, ld == first[:, None], True)).all(axis=1) & (valid.sum(axis=1) >= 2)
    t = df["target"].astype(int).to_numpy()
    inset = (ex == t[:, None]).any(axis=1)
    base = shared & ~inset
    match = (t % 10) == first
    r = df["rating"].to_numpy(dtype=float)
    a = base & match
    b = base & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
