# name: even_set_multiple_of_four_advantage
# description: In all-even sets whose examples are all multiples of 4, yes-rate for targets that are multiples of 4 minus even targets that are not; the model ties x4 to the same 'multiples' salience as x3/x9, so observed < null means it over-credits the x4 rule in even sets (people just use evenness), observed > null means it under-credits it.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    m = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        m[s] = len(ex) >= 2 and all(e % 4 == 0 for e in ex)
    o = sets.map(m).to_numpy().astype(bool)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    a = o & (t % 4 == 0)
    b = o & (t % 2 == 0) & (t % 4 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
