# name: odd_inspan_targets_all_even_sets
# description: Yes rate for odd targets strictly inside the span of sets of 2+ all-even examples that do not all share a last digit; observed below null means people apply an even-number rule more strictly than the model (model over-generalises the range to odd numbers), above means less strictly.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    alleven = np.all(np.isnan(ex) | (ex % 2 == 0), axis=1)
    last = np.where(np.isnan(ex), np.nan, ex % 10)
    samelast = np.nanmax(last, axis=1) == np.nanmin(last, axis=1)
    sel = (k >= 2) & alleven & ~samelast & (t % 2 == 1) & (t > lo) & (t < hi)
    if sel.sum() == 0:
        return 0.0
    return float(r[sel].mean())
