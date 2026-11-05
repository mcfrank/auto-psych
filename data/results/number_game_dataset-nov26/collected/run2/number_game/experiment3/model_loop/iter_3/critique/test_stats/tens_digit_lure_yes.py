# name: tens_digit_lure_yes
# description: Yes rate on multi-example sets whose examples all share a final digit d, for targets whose tens digit is d but whose final digit is not (e.g. 84,94,34 -> 40-49); observed < null means the model over-generalizes a shared final digit to "contains the digit" / tens-position lures.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    dig = np.full(len(u), -1)
    for i, s in enumerate(u):
        ex = [int(x) for x in s.split(",")]
        fd = {e % 10 for e in ex}
        if len(ex) >= 2 and len(fd) == 1:
            dig[i] = fd.pop()
    d = dig[inv]
    t = df["target"].astype(int).values
    m = (d >= 0) & (t >= 10) & (t <= 99) & ((t // 10) == d) & ((t % 10) != d)
    r = df["rating"].astype(float).values
    return float(r[m].mean()) if m.sum() > 0 else 0.0
