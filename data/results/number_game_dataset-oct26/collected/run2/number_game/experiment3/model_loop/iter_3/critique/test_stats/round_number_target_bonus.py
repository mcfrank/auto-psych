# name: round_number_target_bonus
# description: Among sets with no example ending in 0, yes-rate for non-example targets that are multiples of 10 minus non-example targets within 1-2 of a multiple of 10; observed > null means people favour round numbers as program outputs, which the model (final-digit similarity only to shown examples) under-produces.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(ex)), k - 1]
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    no_zero = ~np.any(np.nan_to_num(ex, nan=1.0) % 10 == 0, axis=1)
    r = t % 10
    m = no_zero & ~is_ex
    return rate(m & (r == 0)) - rate(m & ((r == 1) | (r == 2) | (r == 8) | (r == 9)))
