# name: single_digit_target_yes_two_digit_sets
# description: For sets whose examples are all 20 or above, yes rate for single-digit targets (1-9) minus yes rate for other non-example targets; observed above null means people extend to small numbers more than the model, below null means small numbers are rejected more than the model predicts.
def test_statistic(df):
    s = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    cache = {}
    for x in set(s):
        v = [int(a) for a in x.split(",")]
        cache[x] = (set(v), min(v) >= 20)
    ex = np.array([ti in cache[x][0] for ti, x in zip(t, s)])
    big = np.array([cache[x][1] for x in s])
    small = t <= 9
    a = y[big & small]
    b = y[big & ~small & ~ex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
