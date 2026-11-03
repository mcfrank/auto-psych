# name: round_target_bias
# description: Among sets containing no multiple of 5 and targets not in the set, yes rate for targets that are multiples of 10 minus yes rate for other targets; observed > null means people favour round-number targets beyond what the examples license (a bias the model lacks), observed < null means they reject them more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)  # (rows, 4), NaN padded
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    in_set = np.any(E == t[:, None], axis=1)
    no5 = ~np.any(np.mod(E, 5) == 0, axis=1)
    rnd = (t % 10) == 0
    a = no5 & ~in_set & rnd; b = no5 & ~in_set & ~rnd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
