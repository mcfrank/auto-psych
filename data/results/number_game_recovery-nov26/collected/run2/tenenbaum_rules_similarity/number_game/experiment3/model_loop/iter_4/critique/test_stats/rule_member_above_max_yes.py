# name: rule_member_above_max_yes
# description: Among sets of 2+ distinct examples all fitting a rule with <=20 members (squares, cubes, powers, multiples of 5-12, ends in d), yes rate on targets in the smallest such rule that lie above the largest example; observed above null means the model's ratio ceiling wrongly suppresses rule-based upward extrapolation (below null = ceiling too lenient for rules).
def test_statistic(df):
    N = np.arange(1, 101)
    rules = [set(n * n for n in range(1, 11)), set(n ** 3 for n in range(1, 5))]
    for m in range(2, 11):
        rules.append(set(m ** k for k in range(1, 8) if m ** k <= 100))
    for m in range(5, 13):
        rules.append(set(range(m, 101, m)))
    for dgt in range(10):
        rules.append(set(int(n) for n in N if n % 10 == dgt))
    rules = [r for r in rules if len(r) <= 20]
    rules.sort(key=len)
    info = {}
    for s in df["number_set"].astype(str).unique():
        ex = set(int(x) for x in s.split(","))
        mem = None
        if len(ex) >= 2:
            for r in rules:
                if ex <= r:
                    mem = r
                    break
        info[s] = (mem, max(ex))
    ns = df["number_set"].astype(str).values
    tg = df["target"].astype(int).values
    sel = np.array([info[s][0] is not None and t > info[s][1] and t in info[s][0] for s, t in zip(ns, tg)])
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].values[sel].mean())
