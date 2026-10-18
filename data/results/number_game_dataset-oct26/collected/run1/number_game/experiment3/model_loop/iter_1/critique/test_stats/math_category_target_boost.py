# name: math_category_target_boost
# description: Among sets of >=2 examples that all lie in a named category (powers of 2, perfect squares, or cubes), yes-rate for non-example targets in that category minus yes-rate for non-example targets outside it; observed > null means the model (no such concepts) under-generalises to category members.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    t = df["target"].astype(int)
    cats = [set(2 ** k for k in range(0, 7)), set(i * i for i in range(1, 11)), set(i ** 3 for i in range(1, 5))]
    mem = {}
    for s in ns.unique():
        xs = [int(x) for x in s.split(",")]
        if len(xs) < 2:
            continue
        c = [C for C in cats if all(x in C for x in xs)]
        if c:
            mem[s] = (set().union(*c), set(xs))
    pairs = pd.DataFrame({"ns": ns, "t": t}).drop_duplicates()
    lab = {}
    for a, b in zip(pairs["ns"], pairs["t"]):
        if a in mem:
            U, X = mem[a]
            lab[(a, b)] = np.nan if b in X else (1.0 if b in U else 0.0)
    key = pd.Series(list(zip(ns, t)), index=df.index)
    L = key.map(lambda k: lab.get(k, np.nan)) if len(lab) else pd.Series(np.nan, index=df.index)
    r = df["rating"].astype(float)
    a = r[L == 1.0]
    b = r[L == 0.0]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
