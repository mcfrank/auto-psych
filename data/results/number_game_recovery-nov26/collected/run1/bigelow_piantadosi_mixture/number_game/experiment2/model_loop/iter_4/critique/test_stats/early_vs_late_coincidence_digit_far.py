# name: early_vs_late_coincidence_digit_far
# description: Sets of >=3 examples in which some last digit d is shared by >=2 but not all examples: yes-rate on non-example targets ending in d and >10 from every example when the first two examples share d, minus when the sharing only appears at the 3rd/4th example; observed above null_mean means an early-noticed coincidence drives digit generalisation more than the model's rethink mechanism predicts (order effect too weak), below means the model's order effect is too strong.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        xs = [int(v) for v in s.split(",")]
        lab = (-1, -1)
        if len(xs) >= 3:
            ld = [x % 10 for x in xs]
            for d in range(10):
                c = ld.count(d)
                if 2 <= c < len(xs):
                    lab = (d, 1 if (ld[0] == d and ld[1] == d) else 0)
                    break
        info[s] = lab
    lab = sets.map(info)
    d = np.array([v[0] for v in lab.values]); early = np.array([v[1] for v in lab.values])
    t = df["target"].astype(int).values
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").values
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (d >= 0) & (t % 10 == d) & (dist > 10)
    r = df["rating"].values.astype(float)
    a = m & (early == 1); b = m & (early == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
