# name: even_inspan_targets_all_odd_sets
# description: Yes rate for even targets strictly inside the span of sets whose 3-4 examples are all odd; observed above null means the model's odd-number rule suppresses in-range generalisation more than people do (people treat such sets as ranges).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    allodd = np.all(np.isnan(ex) | (ex % 2 == 1), axis=1)
    sel = (k >= 3) & allodd & (t % 2 == 0) & (t > lo) & (t < hi)
    if sel.sum() == 0:
        return 0.0
    return float(r[sel].mean())
