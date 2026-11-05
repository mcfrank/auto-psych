# name: rule_member_out_minus_in_range
# description: For sets of >=3 examples consistent with a parity or multiples-of-m (3-12) rule, among non-example targets that belong to the smallest such consistent rule, yes rate outside the examples' min-max range minus inside; the model's rule/interval mixture predicts a negative gap, so observed above null means people extend broad arithmetic rules beyond the example range as readily as within it (e.g. 63,27,99,33 -> 3; 69,9,39,21 -> 72-96).
def test_statistic(df):
    N = np.arange(1, 101)
    rules = [N % 2 == 0, N % 2 == 1] + [N % m == 0 for m in range(3, 13)]
    M = np.array(rules); sizes = M.sum(axis=1)
    s_arr = df["number_set"].astype(str).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    ins = np.zeros(len(df), bool); out = np.zeros(len(df), bool)
    for s in np.unique(s_arr):
        e = np.array([int(x) for x in s.split(",")])
        if len(e) < 3:
            continue
        ok = M[:, e - 1].all(axis=1)
        if not ok.any():
            continue
        i = np.where(ok)[0][np.argmin(sizes[ok])]
        idx = (s_arr == s) & M[i][t - 1] & ~np.isin(t, e)
        inr = (t >= e.min()) & (t <= e.max())
        ins |= idx & inr; out |= idx & ~inr
    if ins.sum() == 0 or out.sum() == 0:
        return 0.0
    return float(r[out].mean() - r[ins].mean())
