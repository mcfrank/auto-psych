# name: rule_consistent_outside_yes
# description: For multi-example sets that all fit some small arithmetic rule (excluding plain parity/primes/digit-contains), yes-rate on outside-range targets that belong to a fitting rule; observed below null means people generalize rules beyond the example range less than the model's rule weight implies (above: more).
def _rule_sets():
    N = np.arange(1, 101)
    rules = [[2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81], [10, 100]]
    rules += [[i for i in N if i % k == 0] for k in range(3, 13)]
    rules += [[i for i in N if i % 10 == d] for d in range(1, 10)]
    rules += [[1, 4, 9, 16, 25, 36, 49, 64, 81, 100], [1, 8, 27, 64]]
    return [set(int(x) for x in q) for q in rules]


def _rule_member(df, sets, t):
    rules = _rule_sets()
    fit = {}
    for s, v in sets.items():
        ok = [q for q in rules if all(x in q for x in v)]
        fit[s] = set().union(*ok) if (len(v) >= 2 and ok) else None
    has = df["number_set"].map(lambda s: fit[s] is not None).to_numpy()
    mem = np.zeros(len(df), dtype=bool)
    for s, u in fit.items():
        if u is not None:
            m = (df["number_set"] == s).to_numpy()
            mem[m] = np.isin(t[m], list(u))
    return has, mem

def _feat(df):
    sets = {s: [int(x) for x in str(s).split(",")] for s in df["number_set"].unique()}
    ex = df["number_set"].map(sets)
    t = df["target"].astype(int).to_numpy()
    lo = df["number_set"].map({s: min(v) for s, v in sets.items()}).to_numpy()
    hi = df["number_set"].map({s: max(v) for s, v in sets.items()}).to_numpy()
    n = df["number_set"].map({s: len(v) for s, v in sets.items()}).to_numpy()
    arr = np.full((len(df), 4), np.nan)
    for s, v in sets.items():
        m = (df["number_set"] == s).to_numpy()
        arr[m, :len(v)] = v
    dist = np.nanmin(np.abs(arr - t[:, None]), axis=1)
    near = arr[np.arange(len(df)), np.nanargmin(np.abs(arr - t[:, None]), axis=1)]
    return sets, t, lo, hi, n, arr, dist, near, df["rating"].to_numpy().astype(float)


def _m(x):
    return float(np.mean(x)) if len(x) else 0.0


def test_statistic(df):
    sets, t, lo, hi, n, arr, dist, near, r = _feat(df)
    has, mem = _rule_member(df, sets, t)
    return _m(r[has & mem & ((t < lo) | (t > hi))])