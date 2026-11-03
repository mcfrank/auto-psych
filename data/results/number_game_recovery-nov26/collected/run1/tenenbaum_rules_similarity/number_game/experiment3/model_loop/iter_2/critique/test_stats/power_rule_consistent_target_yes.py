# name: power_rule_consistent_target_yes
# description: Among multi-example sets whose examples all lie in one power-type rule (squares, cubes, powers of 2, powers of 3), yes rate for targets that also belong to such a shared rule; observed below null means the model over-generalises mathematical power rules (people endorse rule-consistent targets less than predicted), above means it under-generalises them.
def test_statistic(df):
    rules = [set(k * k for k in range(1, 11)), {1, 8, 27, 64}, {1, 2, 4, 8, 16, 32, 64}, {1, 3, 9, 27, 81}]
    mem = np.zeros((len(rules), 101), bool)
    for j, r in enumerate(rules):
        mem[j, list(r)] = True
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    codes = pd.Categorical(sets, categories=uniq).codes
    tab = np.zeros((len(uniq), len(rules)), bool)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")]
        if len(v) > 1:
            tab[i] = [all(e in r for e in v) for r in rules]
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    sr = tab[codes]
    m = np.any(sr & mem[:, t].T, axis=1)
    return float(y[m].mean()) if m.any() else 0.0
