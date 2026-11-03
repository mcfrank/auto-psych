# name: first_vs_last_example_proximity_gap
# description: For multi-example sets, yes rate on non-example targets within 10 of the first example (and >10 from the last) minus those within 10 of the last example (and >10 from the first); observed below null_mean means people weight later examples more (recency) than the model's first-example sticky guess predicts, above means stronger primacy.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n_ex = np.sum(~np.isnan(E), axis=1)
    last = E[np.arange(len(E)), np.maximum(n_ex - 1, 0)]
    d1 = np.abs(E[:, 0] - t)
    dl = np.abs(last - t)
    notex = ~np.any(E == t[:, None], axis=1)
    multi = n_ex >= 2
    a = multi & notex & (d1 <= 10) & (dl > 10)
    b = multi & notex & (dl <= 10) & (d1 > 10)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
