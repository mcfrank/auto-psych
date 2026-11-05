# name: digit_rule_violation_proximity_contrast
# description: In 2-4-example sets whose examples all share a final digit, yes-rate for wrong-final-digit targets within 10 of some example minus that for wrong-digit targets more than 20 from every example; observed < null means people reject digit-rule violators flatly regardless of distance while the model still lets near violators ride on magnitude similarity (observed > null: people's violation rejection is more distance-graded than the model's).
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    dig = s % 10
    same_dig = (dig.max(axis=1) == dig.min(axis=1)).to_numpy() & (n >= 2)
    viol = same_dig & (t % 10 != dig[0].to_numpy())
    mind = np.nanmin(np.abs(s.to_numpy(dtype=float) - t[:, None]), axis=1)
    r = df["rating"].to_numpy(dtype=float)
    a = viol & (mind <= 10)
    b = viol & (mind > 20)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
