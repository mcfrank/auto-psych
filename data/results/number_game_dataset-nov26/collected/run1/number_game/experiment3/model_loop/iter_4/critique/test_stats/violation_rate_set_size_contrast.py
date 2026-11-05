# name: violation_rate_set_size_contrast
# description: Yes-rate for coincidence-rule-breaking targets (wrong final digit / not multiple of 5 / wrong parity, by the set's most specific shared coincidence) in 4-example coherent sets minus the same in 2-example coherent sets; observed > null (less negative) means rejection of rule-breakers barely strengthens with more examples while the model's coincidence-scaled slope drives 4-example violators much closer to zero, observed < null means people strengthen it more than the model.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    e0 = s[0].to_numpy().astype(int)
    dig = s % 10
    same_dig = (dig.max(axis=1) == dig.min(axis=1)).to_numpy()
    f5 = ((s % 5 == 0) | s.isna()).all(axis=1).to_numpy()
    par = s % 2
    same_par = (par.max(axis=1) == par.min(axis=1)).to_numpy()
    viol = np.where(same_dig, t % 10 != e0 % 10,
           np.where(f5, t % 5 != 0, np.where(same_par, t % 2 != e0 % 2, False))) & (n >= 2)
    r = df["rating"].to_numpy(dtype=float)
    a = viol & (n == 4)
    b = viol & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
