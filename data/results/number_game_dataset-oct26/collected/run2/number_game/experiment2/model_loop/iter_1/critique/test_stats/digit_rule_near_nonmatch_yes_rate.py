# name: digit_rule_near_nonmatch_yes_rate
# description: In sets of 2+ examples sharing one final digit, yes-rate for non-example targets within 3 of some example whose final digit does NOT match; observed above the null means people still generalize by size to near neighbours despite the "ends in d" rule (the model's rule over-suppresses them), below means people reject them even more than the model.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    dig = {}
    for s in sets:
        ex = [int(v) for v in s.split(",")]
        ds = set(v % 10 for v in ex)
        dig[s] = (ex[0] % 10) if (len(ex) >= 2 and len(ds) == 1) else -1
    d = ns.map(dig).values.astype(int)
    exm = {s: np.array([int(v) for v in s.split(",")] + [-1000] * (4 - len(s.split(",")))) for s in sets}
    E = np.stack(ns.map(exm).values)[:, :4]
    t = df["target"].astype(int).values
    mind = np.abs(E - t[:, None]).min(1)
    sel = (d >= 0) & (t % 10 != d) & (mind >= 1) & (mind <= 3)
    y = df["rating"].astype(float).values
    if sel.sum() == 0:
        return 0.0
    return float(y[sel].mean())
