# name: round_target_minus_other_no5_sets
# description: On sets with no example divisible by 5, yes rate for non-example targets that are multiples of 10 minus yes rate for targets not divisible by 5; observed above null means the model under-endorses round numbers (a round-number bias it lacks), below means it over-endorses them.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    no5 = np.all(np.isnan(E) | (np.mod(E, 5) != 0), axis=1)
    a = no5 & (~is_ex) & (t % 10 == 0)
    b = no5 & (~is_ex) & (t % 5 != 0)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
