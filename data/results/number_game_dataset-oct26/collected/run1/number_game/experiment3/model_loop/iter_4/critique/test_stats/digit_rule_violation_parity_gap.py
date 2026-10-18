# name: digit_rule_violation_parity_gap
# description: Among sets whose examples all share one final digit, yes-rate on digit-breaking targets of the same parity as the examples minus on those of opposite parity; observed > null means people still credit parity under an explicit digit rule more than the model's rule-pre-empted parity credit allows; observed < null means parity matters even less than the model assumes once a digit rule is evident.
def test_statistic(df):
    X = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy(float)
    valid = ~np.isnan(X)
    n = valid.sum(1)
    d = np.where(valid, np.mod(np.nan_to_num(X), 10), np.nan)
    dmin = np.nanmin(d, axis=1); dmax = np.nanmax(d, axis=1)
    viol = (n >= 2) & (dmin == dmax) & ((t % 10) != dmin)
    same_par = (t % 2) == np.mod(dmin, 2)
    a = viol & same_par; b = viol & ~same_par
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
