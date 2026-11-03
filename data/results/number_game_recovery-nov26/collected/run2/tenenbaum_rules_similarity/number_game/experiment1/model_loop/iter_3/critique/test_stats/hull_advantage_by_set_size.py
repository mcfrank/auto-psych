# name: hull_advantage_by_set_size
# description: (Yes-rate inside minus outside the example range) for 4-example sets minus the same contrast for 2-3-example sets, non-member targets only; observed below null means people's range-filling advantage does not grow (or shrinks) with more examples as the model's size-principle intervals predict, above null means it grows more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    member = np.nansum(ex == t[:, None], axis=1) > 0
    inside = (t > lo) & (t < hi)
    def contrast(mask):
        a = mask & inside & ~member; b = mask & ~inside & ~member
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return r[a].mean() - r[b].mean()
    return float(contrast(n == 4) - contrast((n == 2) | (n == 3)))
