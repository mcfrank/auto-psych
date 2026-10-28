# name: shared_last_digit_far_target
# description: For 2-4 example sets whose examples all share one last digit, yes-rate for targets with that same last digit lying more than 10 outside the set's range; observed < null means people extend a last-digit rule to far targets less than the model's rule posterior predicts (observed > null: more).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    ld = np.mod(E, 10)
    same = (n >= 2) & (np.nanmin(ld, axis=1) == np.nanmax(ld, axis=1))
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    far = (t < lo - 10) | (t > hi + 10)
    m = same & far & (np.mod(t, 10) == np.nanmin(ld, axis=1))
    return float(y[m].mean()) if m.any() else 0.0
