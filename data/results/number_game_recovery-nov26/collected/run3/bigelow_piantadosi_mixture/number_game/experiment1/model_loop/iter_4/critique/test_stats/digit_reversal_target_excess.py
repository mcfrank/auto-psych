# name: digit_reversal_target_excess
# description: Yes rate for targets that are the digit reversal of an example (e.g. 13 -> 31) minus the yes rate of other non-example targets in the same sets; observed > null means people generalise to digit-reversed numbers more than the model's rule/interval concepts predict.
def test_statistic(df):
    sets = df["number_set"].unique()
    revs = {}
    exs = {}
    for s in sets:
        l = [int(x) for x in s.split(",")]
        exs[s] = set(l)
        revs[s] = {int(str(e)[::-1]) for e in l if e >= 10 and e % 10 != 0 and str(e) != str(e)[::-1]}
    items = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    is_rev = np.array([t in revs[s] for s, t in zip(items["number_set"], items["target"])])
    is_ex = np.array([t in exs[s] for s, t in zip(items["number_set"], items["target"])])
    has = items["number_set"].map(lambda s: len(revs[s]) > 0).to_numpy()
    a = items["rating"][is_rev & ~is_ex]
    b = items["rating"][has & ~is_rev & ~is_ex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
