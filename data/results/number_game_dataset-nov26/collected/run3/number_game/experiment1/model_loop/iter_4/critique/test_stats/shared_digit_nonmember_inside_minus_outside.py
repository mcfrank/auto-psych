# name: shared_digit_nonmember_inside_minus_outside
# description: Among unsorted (not monotone) sets of >=3 examples that all share a last digit, yes rate for targets lacking that digit that lie inside the examples' min-max range minus those outside it; the model's interval component predicts a positive gap, so observed below null means people reject in-range non-members as firmly as out-of-range ones (range adds nothing once a digit rule is clear).
def test_statistic(df):
    s_arr = df["number_set"].astype(str).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    ins = np.zeros(len(df), bool); out = np.zeros(len(df), bool)
    for s in np.unique(s_arr):
        e = np.array([int(x) for x in s.split(",")])
        if len(e) < 3 or len(set(e % 10)) != 1:
            continue
        d = np.diff(e)
        if (d > 0).all() or (d < 0).all():
            continue
        idx = (s_arr == s) & (t % 10 != e[0] % 10)
        inr = (t >= e.min()) & (t <= e.max())
        ins |= idx & inr; out |= idx & ~inr
    if ins.sum() == 0 or out.sum() == 0:
        return 0.0
    return float(r[ins].mean() - r[out].mean())
