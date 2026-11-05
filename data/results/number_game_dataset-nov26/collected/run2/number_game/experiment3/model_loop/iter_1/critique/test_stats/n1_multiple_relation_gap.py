# name: n1_multiple_relation_gap
# description: For one-example sets and same-parity targets only, yes rate when target and example are multiples/divisors of each other (target != example, example > 1) minus yes rate otherwise; observed > null means people generalize along the multiplicative relation more than the model (which lacks a 'multiples of the example' concept beyond fixed moduli 3-12) predicts.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = (s.str.count(",") == 0).values
    e = pd.to_numeric(s.where(one, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    ok = one & (e > 1) & (t != e) & (t % 2 == e % 2)
    es = np.where(e > 0, e, 1)
    rel = ((t % es == 0) | (es % t == 0))
    a = ok & rel
    b = ok & ~rel
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
