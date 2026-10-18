# name: all_even_minus_all_odd_sets
# description: Yes rate on sets whose examples are all even minus yes rate on sets whose examples are all odd (any target); observed below null means people generalize more narrowly from even sets (e.g. inferring multiples rules) than the parity-symmetric digit model predicts, above means more broadly.
def test_statistic(df):
    s = df["number_set"].astype(str)
    uniq = s.unique()
    kind = {}
    for u in uniq:
        ps = set(int(v) % 2 for v in u.split(","))
        kind[u] = 0 if ps == {0} else (1 if ps == {1} else -1)
    k = s.map(kind).astype(int)
    r = df["rating"].astype(float)
    a = r[k == 0]
    b = r[k == 1]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
