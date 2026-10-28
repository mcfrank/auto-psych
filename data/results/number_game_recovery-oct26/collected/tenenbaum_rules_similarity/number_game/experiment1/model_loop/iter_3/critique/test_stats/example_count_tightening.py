# name: example_count_tightening
# description: Among non-example targets within 20 of the nearest example, yes rate for single-example sets minus sets of 3-4 examples; the size principle makes generalization tighten with more examples, so observed above null_mean means people tighten more with extra examples than the model, below means less.
def test_statistic(df):
    ex = pd.DataFrame(df["number_set"].astype(str).str.split(",").tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n_ex = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(y[a].mean() - y[b].mean())
    base = (~member) & (dist <= 20)
    return diff(base & (n_ex == 1), base & (n_ex >= 3))
