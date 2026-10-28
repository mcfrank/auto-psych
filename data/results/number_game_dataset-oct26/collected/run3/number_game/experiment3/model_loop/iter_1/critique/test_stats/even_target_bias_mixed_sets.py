# name: even_target_bias_mixed_sets
# description: For sets of 2+ examples containing both odd and even numbers, yes-rate for even non-example targets minus odd ones; observed > null means people favour even targets beyond any rule the examples support (observed < null: favour odd), a stimulus bias the model lacks.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    p = np.mod(E, 2)
    mixed = (n >= 2) & (np.nanmin(p, axis=1) != np.nanmax(p, axis=1))
    notin = ~np.any(E == t[:, None], axis=1)
    ev = np.mod(t, 2) == 0
    a = mixed & notin & ev; b = mixed & notin & ~ev
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
