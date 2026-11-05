# name: coherent_far_consistent_set_size_slope
# description: Among coherent 2-4-example sets (same final digit, all multiples of 5, or same parity), yes-rate for rule-consistent targets more than 20 from every example in 4-example sets minus in 2-example sets; observed > null means people extend the rule to far targets more with more examples than the model (whose averaging penalty and distance gradient dominate), observed < null means the model over-extends it with set size.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    t = df["target"].astype(int).values
    n = s.notna().sum(axis=1).values
    par = s % 2
    same_par = (par.max(axis=1) == par.min(axis=1)).values
    dig = s % 10
    same_dig = (dig.max(axis=1) == dig.min(axis=1)).values
    f5 = ((s % 5 == 0) | s.isna()).all(axis=1).values
    e0 = s[0].values.astype(int)
    viol = np.where(same_dig, (t % 10) != (e0 % 10),
                    np.where(f5, (t % 5) != 0, (t % 2) != (e0 % 2)))
    dmin = np.nanmin(np.abs(s.values - t[:, None]), axis=1)
    base = (same_par | same_dig | f5) & ~viol & (dmin > 20)
    r = df["rating"].values
    a = base & (n == 4)
    b = base & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
