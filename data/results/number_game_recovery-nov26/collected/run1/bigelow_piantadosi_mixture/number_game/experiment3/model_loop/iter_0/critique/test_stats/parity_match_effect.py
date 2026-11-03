# name: parity_match_effect
# description: In sets of >=2 examples that all share a parity, yes rate for non-example targets of that parity minus those of the opposite parity; observed > model means the model under-weights parity (odd/even) generalization, observed < model means it over-weights it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    tg = df["target"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    n_odd = np.where(valid, np.mod(ex, 2), 0).sum(axis=1)
    same = (n >= 2) & ((n_odd == 0) | (n_odd == n))
    set_par = (n_odd > 0).astype(float)
    inset = (ex == tg[:, None]).any(axis=1)
    match = np.mod(tg, 2) == set_par
    r = df["rating"].astype(float).to_numpy()
    a = same & ~inset & match
    b = same & ~inset & ~match
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
