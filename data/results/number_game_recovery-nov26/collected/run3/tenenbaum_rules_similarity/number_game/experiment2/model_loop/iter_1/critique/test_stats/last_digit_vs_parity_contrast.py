# name: last_digit_vs_parity_contrast
# description: For sets of >=2 examples sharing one last digit, among non-example targets of the same parity as the examples, yes rate when the target shares that last digit minus when it does not; observed > null means people use the "ends in d" rule more strongly (relative to odd/even) than the model's family prior predicts, observed < null less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    d = np.mod(E, 10)
    dmin, dmax = np.nanmin(d, axis=1), np.nanmax(d, axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    base = (n >= 2) & (dmin == dmax) & ~in_set & (np.mod(t, 2) == np.mod(dmin, 2))
    same = base & (np.mod(t, 10) == dmin)
    diff = base & (np.mod(t, 10) != dmin)
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[diff].mean())
