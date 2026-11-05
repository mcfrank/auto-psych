# name: in_range_opposite_parity_contrast
# description: For sets of >=3 examples all of one parity that do not all share a last digit, yes rate for in-range (between min and max example) non-example targets of the opposite parity minus same parity; the model's parity-blind intervals predict a small gap, so observed below null means people restrict range generalization to the examples' parity (e.g. 23,35,95,83 rejects even 30-36).
def test_statistic(df):
    s_arr = df["number_set"].astype(str).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    opp = np.zeros(len(df), bool); same = np.zeros(len(df), bool)
    for s in np.unique(s_arr):
        e = np.array([int(x) for x in s.split(",")])
        if len(e) < 3 or len(set(e % 2)) != 1 or len(set(e % 10)) == 1:
            continue
        idx = (s_arr == s) & (t >= e.min()) & (t <= e.max()) & ~np.isin(t, e)
        sp = (t % 2) == (e[0] % 2)
        same |= idx & sp; opp |= idx & ~sp
    if opp.sum() == 0 or same.sum() == 0:
        return 0.0
    return float(r[opp].mean() - r[same].mean())
