# name: odd_set_parity_contrast
# description: For sets of 3-4 examples that are all odd but do NOT all share a last digit (e.g. 33,17,5,9; 69,9,39,21), yes rate on odd non-example targets minus even targets; observed above null_mean means people use the broad "odd numbers" rule more than the model predicts, below means the model over-weights the odd rule.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    odd = np.all((np.mod(ex, 2) == 1) | np.isnan(ex), axis=1)
    ld = np.mod(ex, 10)
    same = np.all((ld == ld[:, :1]) | np.isnan(ld), axis=1)
    sel = (n >= 3) & odd & ~same & ~np.any(ex == t[:, None], axis=1)
    a, b = sel & (t % 2 == 1), sel & (t % 2 == 0)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
