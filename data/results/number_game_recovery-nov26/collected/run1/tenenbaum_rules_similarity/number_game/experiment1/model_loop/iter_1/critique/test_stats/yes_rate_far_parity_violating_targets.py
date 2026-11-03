# name: yes_rate_far_parity_violating_targets
# description: Yes-rate among targets that lie >10 outside the example range AND have the opposite parity of an all-same-parity example set; observed > null means the model under-produces baseline/lapse "yes" answers for concept-inconsistent targets (it has no noise term).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(int).values
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    allsame = np.nanmin(par, axis=1) == np.nanmax(par, axis=1)
    mask = allsame & ((t % 2) != np.nanmax(par, axis=1)) & ((t < lo - 10) | (t > hi + 10))
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
