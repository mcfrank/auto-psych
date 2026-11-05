# name: round_number_target_lift
# description: Among non-example targets for sets whose examples are NOT all multiples of 10, yes-rate on targets that are multiples of 10 minus yes-rate on other targets; the model credits multiples of 10 only when all examples share it, so observed above null means a round-number endorsement bias the model misses, below means people shun round targets.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    isex = np.any(E == t[:, None], axis=1)
    allten = np.all(np.isnan(E) | (np.mod(E, 10) == 0), axis=1)
    ok = ~isex & ~allten
    rnd = np.mod(t, 10) == 0
    a = ok & rnd; b = ok & ~rnd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
