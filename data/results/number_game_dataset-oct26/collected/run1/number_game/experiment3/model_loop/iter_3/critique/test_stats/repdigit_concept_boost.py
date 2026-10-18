# name: repdigit_concept_boost
# description: In sets containing at least one repdigit example (11,22,...,99), yes-rate on repdigit targets minus yes-rate on non-repdigit targets that share a final digit with some example (examples themselves excluded); observed > null means the model under-produces a "same-digit numbers" concept, observed < null that it over-credits it.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    cats = pd.unique(sets)
    info = {}
    for s in cats:
        xs = [int(x) for x in s.split(",")]
        info[s] = (any(x % 11 == 0 and 11 <= x <= 99 for x in xs), set(xs), {x % 10 for x in xs})
    sv = sets.values
    t = df["target"].astype(int).values
    has = np.array([info[s][0] for s in sv])
    is_ex = np.array([ti in info[s][1] for s, ti in zip(sv, t)])
    dm = np.array([(ti % 10) in info[s][2] for s, ti in zip(sv, t)])
    rep = (t % 11 == 0) & (t >= 11) & (t <= 99)
    r = df["rating"].values.astype(float)
    a = has & ~is_ex & rep
    b = has & ~is_ex & ~rep & dm
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
