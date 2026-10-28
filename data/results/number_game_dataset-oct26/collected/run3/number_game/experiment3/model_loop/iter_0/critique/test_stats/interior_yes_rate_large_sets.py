# name: interior_yes_rate_large_sets
# description: Yes-rate for targets strictly inside the [min,max] range of 3-4 example sets (not themselves examples); observed < null means people fill in the gaps between examples less than the model's interval hypotheses imply (observed > null: more).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    ex = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    E = ex[codes]  # (rows, 4) examples, NaN-padded, in shown order
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    inset = np.any(E == t[:, None], axis=1)
    m = (n >= 3) & (t > lo) & (t < hi) & ~inset
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
