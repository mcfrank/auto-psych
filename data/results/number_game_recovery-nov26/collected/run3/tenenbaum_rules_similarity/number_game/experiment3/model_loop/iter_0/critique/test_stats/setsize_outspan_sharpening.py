# name: setsize_outspan_sharpening
# description: Yes rate on targets outside the set's [min, max] span for 4-example sets minus that for 2-example sets; observed below null means people narrow generalisation with more examples more strongly than the model's size principle predicts, above null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = (~np.isnan(ex)).sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    out = (t < np.nanmin(ex, axis=1)) | (t > np.nanmax(ex, axis=1))
    r = df["rating"].to_numpy().astype(float)
    a = out & (n == 4)
    b = out & (n == 2)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
