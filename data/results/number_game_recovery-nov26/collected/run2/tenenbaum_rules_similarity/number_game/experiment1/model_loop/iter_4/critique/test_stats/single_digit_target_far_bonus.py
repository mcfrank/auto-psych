# name: single_digit_target_far_bonus
# description: Among targets more than 10 away from every example and not in the set, yes-rate on single-digit targets (1-9) minus targets 10-100; observed above null means people favour small numbers beyond what the concept hypotheses predict (a magnitude/small-number prior the model lacks), below null means they disfavour them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    base = d > 10
    r = df["rating"].to_numpy(dtype=float)
    a = base & (t < 10)
    b = base & (t >= 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
