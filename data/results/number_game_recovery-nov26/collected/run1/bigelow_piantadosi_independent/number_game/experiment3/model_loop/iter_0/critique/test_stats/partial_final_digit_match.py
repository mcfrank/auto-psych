# name: partial_final_digit_match
# description: In sets of 3-4 examples with at least two distinct final digits, yes-rate for non-example targets sharing their final digit with exactly one example; observed above the null means people generalize from single examples' final digits ("seen final digits") more than the model's one-rule-plus-exception account allows.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    D = np.where(np.isnan(E), -1.0, np.mod(E, 10))
    # distinct final digits per row (vectorised over the 4 columns)
    Ds = np.sort(D, axis=1)
    distinct = (Ds[:, 0] >= 0).astype(int) + np.sum((np.diff(Ds, axis=1) != 0) & (Ds[:, 1:] >= 0), axis=1)
    matches = np.sum(D == np.mod(t, 10)[:, None], axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    m = (n >= 3) & (distinct >= 2) & (matches == 1) & ~is_ex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
