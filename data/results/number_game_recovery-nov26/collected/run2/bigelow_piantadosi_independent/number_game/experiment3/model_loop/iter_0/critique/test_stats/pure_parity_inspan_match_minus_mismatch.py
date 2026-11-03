# name: pure_parity_inspan_match_minus_mismatch
# description: In 2+-example sets whose examples all share one parity, yes-rate for same-parity non-example targets inside the span minus opposite-parity targets inside the span; observed above null_mean means the parity penalty for in-range targets is larger than the model's constant veto plus inferred concept produce (below means smaller).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    odd = np.where(valid, np.mod(ex, 2) == 1, False).sum(axis=1)
    pure = (n >= 2) & ((odd == 0) | (odd == n))
    par = np.mod(np.nan_to_num(ex[:, 0]), 2)
    t = df["target"].astype(float).to_numpy()
    inset = (ex == t[:, None]).any(axis=1)
    inspan = (t >= np.nanmin(ex, axis=1)) & (t <= np.nanmax(ex, axis=1))
    same = np.mod(t, 2) == par
    base = pure & inspan & ~inset
    r = df["rating"].astype(float).to_numpy()
    a, b = base & same, base & ~same
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
