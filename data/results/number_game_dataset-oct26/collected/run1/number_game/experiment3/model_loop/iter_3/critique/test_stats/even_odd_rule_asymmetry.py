# name: even_odd_rule_asymmetry
# description: Yes-rate on even targets in all-odd multi-example sets minus yes-rate on odd targets in all-even multi-example sets (targets sharing an example's final digit excluded); observed < null means people reject parity-violating targets more symmetrically for odd sets than the model's even-only privileged rule allows (an "odd numbers" rule is also used), observed > null means the even/odd asymmetry is even stronger than modelled.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    cats = pd.unique(sets)
    info = {}
    for s in cats:
        xs = [int(x) for x in s.split(",")]
        par = 0 if all(x % 2 == 0 for x in xs) else (1 if all(x % 2 == 1 for x in xs) else -1)
        info[s] = (par if len(xs) >= 2 else -1, {x % 10 for x in xs})
    sv = sets.values
    t = df["target"].astype(int).values
    par = np.array([info[s][0] for s in sv])
    dm = np.array([(ti % 10) in info[s][1] for s, ti in zip(sv, t)])
    r = df["rating"].values.astype(float)
    a = (par == 1) & (t % 2 == 0) & ~dm
    b = (par == 0) & (t % 2 == 1) & ~dm
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
