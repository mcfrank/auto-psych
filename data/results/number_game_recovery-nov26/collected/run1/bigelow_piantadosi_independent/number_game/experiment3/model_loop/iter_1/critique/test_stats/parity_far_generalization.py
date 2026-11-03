# name: parity_far_generalization
# description: In 2-4 example sets whose examples all share a parity, yes-rate for same-parity non-example targets more than 10 from every example and sharing no final digit with any example, minus the yes-rate for opposite-parity targets meeting the same conditions; observed above the null means people extend a parity rule to distant numbers more than the model, below means the model's parity generalization is too strong.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    Dg = np.where(np.isnan(E), np.nan, np.mod(E, 10))
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    par = np.where(np.isnan(E), np.nan, np.mod(E, 2))
    odd_n = np.nansum(par == 1, axis=1)
    even_n = np.nansum(par == 0, axis=1)
    allp = (n >= 2) & ((odd_n == n) | (even_n == n))
    set_par = np.where(odd_n == n, 1, 0)
    nondig = ~is_ex & ~np.any(Dg == np.mod(t, 10)[:, None], axis=1)
    base = allp & nondig & (dist > 10)
    a = base & (np.mod(t, 2) == set_par)
    b = base & (np.mod(t, 2) != set_par)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
