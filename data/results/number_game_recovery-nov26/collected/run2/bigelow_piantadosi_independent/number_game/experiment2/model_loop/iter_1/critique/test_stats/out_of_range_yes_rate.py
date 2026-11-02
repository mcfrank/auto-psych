# name: out_of_range_yes_rate
# description: For sets of 2+ examples, yes-rate for targets lying outside [min, max] of the examples by more than 5; observed > null means people extend generalization beyond the example range more than the model predicts.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sel = (n >= 2) & ((t < lo - 5) | (t > hi + 5))
    if sel.sum() == 0:
        return 0.0
    return float(r[sel].mean())
