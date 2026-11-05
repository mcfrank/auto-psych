# name: mult5_vs_other_multiple_gap
# description: In sets whose examples share both a multiple-of-5/10 property and some other multiple-of-m property (m in 3..12, e.g. 90,45 share x5 and x9), yes-rate for targets that are multiples of the shared 5/10 (but not of the other shared m) minus targets that are multiples of another shared m but not of the shared 5/10; observed > null means the model's single 'multiples' salience under-differentiates base-10 multiples from other multiples (people strongly prefer x5/x10 rules).
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    a = np.zeros(len(t), bool); b = np.zeros(len(t), bool)
    for s in pd.unique(sets):
        xs = [int(x) for x in s.split(",")]
        if len(xs) < 2:
            continue
        sh = [m for m in range(3, 13) if all(x % m == 0 for x in xs)]
        five = [m for m in sh if m in (5, 10)]
        oth = [m for m in sh if m not in (5, 10)]
        if not five or not oth:
            continue
        idx = sets == s
        h5 = np.zeros(len(t), bool); ho = np.zeros(len(t), bool)
        for m in five:
            h5 |= (t % m == 0)
        for m in oth:
            ho |= (t % m == 0)
        a |= idx & h5 & ~ho
        b |= idx & ho & ~h5
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
