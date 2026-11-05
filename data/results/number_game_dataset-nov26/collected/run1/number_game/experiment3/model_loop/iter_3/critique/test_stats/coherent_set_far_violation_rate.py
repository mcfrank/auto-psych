# name: coherent_set_far_violation_rate
# description: Yes-rate for targets more than 20 from every example that break the set's coincidence (wrong final digit for same-digit sets, else not a multiple of 5 for all-multiple-of-5 sets, else wrong parity for same-parity sets), among 2-4-example coherent sets; observed > null means the model's steepened slope drives these rejections too close to zero (people keep a residual yes floor), observed < null means the model accepts them too often.
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
    m = (n >= 2) & (same_par | same_dig | f5) & viol & (dmin > 20)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
