# name: outlier_tolerant_rule_yes
# description: Among 3-4-example sets where no rule of size <=25 (multiples of 3-50, ends-in-digit, squares, powers of 2/3) fits all examples but one fits all but one, yes rate for non-example targets in that smallest leave-one-out rule; observed above the null means people discount a single outlier example and still follow the rule more than the model's random-example noise allows (below = they abandon the rule more readily).
def test_statistic(df):
    rules = [set(range(m, 101, m)) for m in range(4, 51)]
    rules += [set(range(d if d else 10, 101, 10)) for d in range(10)]
    rules += [{n * n for n in range(1, 11)}, {2, 4, 8, 16, 32, 64}, {3, 9, 27, 81}]
    rules = [r for r in rules if len(r) <= 25]
    rules.sort(key=len)
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        best = None
        if len(ex) >= 3 and not any(all(x in r for x in ex) for r in rules):
            for r in rules:
                if sum(x in r for x in ex) == len(ex) - 1:
                    best = r
                    break
        info[s] = (best, set(ex))
    t = df["target"].astype(int).to_numpy()
    m = np.array([info[s][0] is not None and tt in info[s][0] and tt not in info[s][1]
                  for s, tt in zip(sets, t)])
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
