# name: multiple_of_3_set_contrast
# description: Among multi-example sets whose examples are all multiples of 3 (not all sharing a last digit), yes-rate for non-example targets that are multiples of 3 minus yes-rate for non-example targets that are not; observed above model means people generalise "multiples of 3" more sharply than the model's uniform-within-family multiples prior allows, below means less sharply.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    keep = []
    exsets = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        if len(ex) >= 2 and all(e % 3 == 0 for e in ex) and len(set(e % 10 for e in ex)) > 1:
            keep.append(s)
            exsets[s] = set(ex)
    if not keep:
        return 0.0
    sub = df[sets.isin(keep).to_numpy()]
    t = sub["target"].astype(int).to_numpy()
    ss = sub["number_set"].astype(str).to_numpy()
    inset = np.array([x in exsets[s] for s, x in zip(ss, t)])
    r = sub["rating"].to_numpy(dtype=float)
    m3 = (t % 3 == 0) & ~inset
    n3 = (t % 3 != 0) & ~inset
    if m3.sum() == 0 or n3.sum() == 0:
        return 0.0
    return float(r[m3].mean() - r[n3].mean())
