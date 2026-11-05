# name: shared_digit_yes_rate_diff
# description: Among targets not in the set, yes-rate for targets sharing their last digit with some example minus yes-rate for those sharing no last digit; observed > null means people use surface digit similarity more than the model's rule/range concepts capture.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    sv = sets.values
    share = np.zeros(len(t), bool)
    inset = np.zeros(len(t), bool)
    for s in sets.unique():
        xs = np.array([int(x) for x in s.split(",")])
        idx = np.where(sv == s)[0]
        tt = t[idx]
        share[idx] = np.isin(tt % 10, xs % 10)
        inset[idx] = np.isin(tt, xs)
    a = share & ~inset
    b = ~share & ~inset
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
