# name: math_category_match_contrast
# description: For sets of >=2 examples that all belong to a mathematical category the model does not represent (perfect squares, powers of two, cubes, primes), yes-rate on targets inside that category minus yes-rate on targets outside it; observed > null means people use these categories and the model under-produces category-based generalization.
def test_statistic(df):
    g = df.groupby(["number_set", "target"], sort=False)["rating"].mean().reset_index()
    sq = {i * i for i in range(1, 11)}
    p2 = {2 ** i for i in range(0, 7)}
    cu = {i ** 3 for i in range(1, 5)}
    pr = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}
    cats = [sq, p2, cu, pr]
    inn, out = [], []
    for s, t, r in zip(g["number_set"], g["target"].astype(int), g["rating"]):
        ex = [int(x) for x in str(s).split(",")]
        if len(ex) < 2:
            continue
        hit = [c for c in cats if all(x in c for x in ex)]
        if not hit:
            continue
        (inn if any(t in c for c in hit) else out).append(r)
    if not inn or not out:
        return 0.0
    return float(np.mean(inn) - np.mean(out))
