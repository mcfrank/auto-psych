# name: in_range_nonrule_same_digit_sets
# description: For 3-4 example sets whose examples all share one last digit, yes-rate for targets strictly inside the [min,max] range that do NOT share that last digit; observed > null means people still accept in-range numbers (interval-like generalization) when a sharp digit rule fits, more than the model's rule-dominated posterior allows (observed < null: humans stricter).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    ld = np.mod(E, 10)
    same = (n >= 3) & (np.nanmin(ld, axis=1) == np.nanmax(ld, axis=1))
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    m = same & (t > lo) & (t < hi) & (np.mod(t, 10) != np.nanmin(ld, axis=1))
    return float(y[m].mean()) if m.any() else 0.0
