# name: single_vs_multi_example_near_generalization
# description: Yes rate for targets 1-10 outside the examples' range on 1-example sets minus on 3-4-example sets; the size principle makes the model tighten with more examples, so observed below null_mean means people tighten less (or broaden) with more examples than the model, above means they tighten more.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",")
    ex = pd.DataFrame(sets.tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    n_ex = np.sum(~np.isnan(ex), axis=1)
    outside = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    m = (outside >= 1) & (outside <= 10)
    a = m & (n_ex == 1); b = m & (n_ex >= 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
