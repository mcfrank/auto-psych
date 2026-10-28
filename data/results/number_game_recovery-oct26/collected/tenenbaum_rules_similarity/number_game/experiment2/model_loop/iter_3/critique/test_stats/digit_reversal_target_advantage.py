# name: digit_reversal_target_advantage
# description: Yes-rate on targets that are the digit-reversal of a two-digit non-palindromic example (e.g. 13 after 31) and not themselves examples, minus the yes-rate on other non-example targets in the same sets that share a digit with some example; observed > null means people use digit-permutation similarity the model lacks.
def test_statistic(df):
    exl = df["number_set"].map(lambda s: [int(x) for x in s.split(",")]).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    rev = []
    shared = []
    isex = []
    has_rev_set = []
    for tt, ex in zip(t, exl):
        rs = set(int(str(e)[::-1]) for e in ex if 10 <= e <= 99 and e % 10 != 0 and e % 11 != 0)
        digs = set("".join(str(e) for e in ex))
        rev.append(tt in rs)
        shared.append(bool(set(str(tt)) & digs))
        isex.append(tt in ex)
        has_rev_set.append(len(rs) > 0)
    rev = np.array(rev); shared = np.array(shared); isex = np.array(isex); hs = np.array(has_rev_set)
    a = rev & ~isex
    b = hs & ~rev & ~isex & shared
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
