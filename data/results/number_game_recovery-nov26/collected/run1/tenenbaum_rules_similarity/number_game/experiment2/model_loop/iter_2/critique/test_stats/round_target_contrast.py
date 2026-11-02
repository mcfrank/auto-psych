# name: round_target_contrast
# description: Among sets containing no multiple of 10, yes rate on non-member targets that are multiples of 10 minus yes rate on other non-member targets; observed below null means people avoid round numbers that the examples do not license more than the model (above null: a round-number attraction the model lacks).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    has10 = np.any(np.nan_to_num(ex, nan=1.0) % 10 == 0, axis=1)
    mem = np.any(ex == t[:, None], axis=1)
    base = (~has10) & (~mem)
    a = base & (t % 10 == 0); b = base & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
