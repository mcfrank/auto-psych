# name: multi_set_size_yes_slope
# description: Among multi-example sets, yes-rate on non-example targets for 4-example sets minus 2-example sets; the model has no explicit size principle (only property sharing), so observed below null means people narrow generalization with more examples more than modelled, above means they narrow less (or broaden).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    isex = np.any(E == t[:, None], axis=1)
    a = ~isex & (n == 4); b = ~isex & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
