# name: same_last_digit_set_sharpness
# description: For sets of 3-4 examples that all share one last digit (e.g. 72,92,82,42; 80,70,90,100), yes rate on non-example targets sharing that last digit minus yes rate on all other targets; observed above null_mean means people endorse the "ends in d" rule more sharply than the model, below means the model's rule generalization is too sharp/all-or-none.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    ld = np.mod(ex, 10)
    d0 = ld[:, 0]
    same = np.all((ld == d0[:, None]) | np.isnan(ld), axis=1) & (n >= 3)
    notex = ~np.any(ex == t[:, None], axis=1)
    match = np.mod(t, 10) == d0
    a, b = same & notex & match, same & notex & ~match
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
