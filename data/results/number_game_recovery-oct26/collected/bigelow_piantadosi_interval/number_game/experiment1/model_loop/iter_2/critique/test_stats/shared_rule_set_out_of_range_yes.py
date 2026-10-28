# name: shared_rule_set_out_of_range_yes
# description: Yes rate on non-example targets outside the [min,max] example range, restricted to multi-example sets whose examples all share a last digit or are all multiples of the same n in 3..12; observed above null means people extend a clear arithmetic rule beyond the example range more than the model (too much interval mass), below means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = list(sets.unique())
    info = {}
    for s in uniq:
        a = np.array([int(x) for x in s.split(",")])
        rule = False
        if len(a) >= 2:
            rule = bool(np.all(a % 10 == a[0] % 10)) or any(np.all(a % n == 0) for n in range(3, 13))
        info[s] = (rule, a.min(), a.max(), s)
    rule = sets.map({s: v[0] for s, v in info.items()}).to_numpy().astype(bool)
    lo = sets.map({s: v[1] for s, v in info.items()}).to_numpy()
    hi = sets.map({s: v[2] for s, v in info.items()}).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sel = rule & ((t < lo) | (t > hi))
    return float(r[sel].mean()) if sel.any() else 0.0
