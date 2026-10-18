# name: digit_rule_size_principle
# description: Among multi-example sets whose examples all share one final digit, yes-rate on targets NOT matching that digit for sets of >=3 examples minus for 2-example sets; observed < null means people reject rule-violating targets more strongly as consistent examples accumulate (a size principle) than the model, whose mutual-similarity shortfall is identical for 2 or 4 uniform examples; observed > null the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    cats = pd.unique(sets)
    info = {}
    for s in cats:
        xs = [int(x) for x in s.split(",")]
        u = len(xs) >= 2 and len({x % 10 for x in xs}) == 1
        info[s] = (len(xs) if u else 0, xs[0] % 10)
    sv = sets.values
    t = df["target"].astype(int).values
    n = np.array([info[s][0] for s in sv])
    d = np.array([info[s][1] for s in sv])
    nm = (t % 10) != d
    r = df["rating"].values.astype(float)
    a = (n >= 3) & nm
    b = (n == 2) & nm
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
