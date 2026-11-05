# name: common_factor_nodigit_contrast
# description: Among targets matching no example's final digit and not equal to an example (sets with >=2 examples), yes-rate for targets sharing a divisor >1 with EVERY example minus yes-rate for targets coprime with at least one example; observed < null means the model's coprime penalty over-credits shared factors (people don't use shared divisibility that much), observed > null means people reward shared factors more than modelled.
def test_statistic(df):
    from math import gcd
    it = df.groupby(["number_set", "target"])["rating"].agg(["sum", "count"]).reset_index()
    sa = sc = sb = cb = 0.0
    for s, t, k, c in zip(it["number_set"].values, it["target"].values, it["sum"].values, it["count"].values):
        ex = [int(x) for x in s.split(",")]
        t = int(t)
        if len(ex) < 2 or t in ex or any(t % 10 == e % 10 for e in ex):
            continue
        if all(gcd(t, e) > 1 for e in ex):
            sa += k; sc += c
        else:
            sb += k; cb += c
    if sc == 0 or cb == 0:
        return 0.0
    return float(sa / sc - sb / cb)
