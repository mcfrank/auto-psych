# name: far_rule_member_yes
# description: Among sets of >=2 examples that all belong to a small arithmetic rule (powers of 2-10, squares, cubes, multiples of 3-12; <=34 members), yes rate for non-example targets that are members of such a rule containing all examples and lie >=20 from the nearest example; observed > null means the model's distance hesitance over-discounts far rule members (people extend rules far), observed < null that it under-discounts them.
def test_statistic(df):
    rules = []
    for m in range(2, 11):
        rules.append([m ** k for k in range(1, 8) if m ** k <= 100])
    rules.append([n * n for n in range(1, 11)])
    rules.append([n ** 3 for n in range(1, 5)])
    for m in range(3, 13):
        rules.append(list(range(m, 101, m)))
    M = np.zeros((len(rules), 101), dtype=bool)
    for i, r in enumerate(rules):
        M[i, r] = True
    sets = df["number_set"].astype(str)
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).fillna(0).astype(int).values
    t = df["target"].astype(int).values
    has = ex > 0
    nex = has.sum(1)
    # rule contains all examples: (rows, rules)
    allin = np.ones((len(df), len(rules)), dtype=bool)
    for k in range(4):
        allin &= np.where(has[:, k][:, None], M[:, ex[:, k]].T, True)
    tin = M[:, t].T
    hit = (allin & tin).any(1)
    dist = np.where(has, np.abs(ex - t[:, None]), 999).min(1)
    sel = (nex >= 2) & hit & (dist >= 20)
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].values[sel].mean())
