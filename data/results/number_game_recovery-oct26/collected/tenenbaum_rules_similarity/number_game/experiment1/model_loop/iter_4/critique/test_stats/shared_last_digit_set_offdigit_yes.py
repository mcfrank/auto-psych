# name: shared_last_digit_set_offdigit_yes
# description: For sets of 2+ examples that all share the same final digit (e.g. 46,76,66), yes rate on non-example targets whose final digit differs; observed below null_mean means people commit to the ends-in-d rule more sharply (reject off-digit targets more) than the model, above means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n_ex = valid.sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    ld = np.where(valid, np.mod(ex, 10), np.nan)
    same = (np.nanmax(ld, axis=1) == np.nanmin(ld, axis=1)) & (n_ex >= 2)
    d0 = np.nanmax(ld, axis=1)
    sel = same & (~member) & (np.mod(t, 10) != d0)
    if sel.sum() == 0:
        return 0.0
    return float(y[sel].mean())
