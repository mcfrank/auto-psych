# name: ends5_to_ends0_extension
# description: Sets of >=2 examples all ending in 5: yes-rate for targets ending in 0 minus targets ending in neither 0 nor 5; observed above null_mean means people read such sets as "multiples of 5" more than the model (which favours "ends in 5"), below means the model over-produces the multiples-of-5 reading.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ok = {}
    for s in sets.unique():
        xs = [int(v) for v in s.split(",")]
        ok[s] = len(xs) >= 2 and all(x % 10 == 5 for x in xs)
    sel = sets.map(ok).values.astype(bool)
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    a = sel & (t % 10 == 0); b = sel & (t % 5 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
