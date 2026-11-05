# name: odd_x3_parity_vs_multiple_tradeoff
# description: In sets (>=2 examples) whose examples are all odd and all multiples of 3, yes-rate for odd non-multiples of 3 minus even multiples of 3; observed > null means the model under-weights parity relative to the shared multiple-of-3 rule in these sets (observed < null: over-weights parity).
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",")
    n = ex.str.len().values
    ok = ex.map(lambda l: all(int(x) % 2 == 1 and int(x) % 3 == 0 for x in l)).values & (n >= 2)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    a = ok & (t % 2 == 1) & (t % 3 != 0)
    b = ok & (t % 2 == 0) & (t % 3 == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
