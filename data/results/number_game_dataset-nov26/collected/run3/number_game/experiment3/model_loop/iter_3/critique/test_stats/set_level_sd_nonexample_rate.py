# name: set_level_sd_nonexample_rate
# description: Standard deviation across 2-4 example sets of each set's yes rate for non-example targets; observed above null means the model is too homogeneous across sets (people generalize much more broadly from some sets and much more narrowly from others than the model's concept posterior implies).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    Ez = np.where(valid, E, 0.0)
    lo = np.where(valid, E, np.inf).min(axis=1)
    hi = np.where(valid, E, -np.inf).max(axis=1)
    is_ex = (valid & (E == t[:, None])).any(axis=1)
    odd = (np.where(valid, Ez % 2, 1) == 1).all(axis=1)
    even = (np.where(valid, Ez % 2, 0) == 0).all(axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    m = (n >= 2) & ~is_ex
    s = pd.Series(y[m]).groupby(df["number_set"].to_numpy()[m]).mean()
    return float(s.std()) if len(s) > 1 else 0.0
