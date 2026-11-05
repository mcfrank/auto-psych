# name: shared_final_digit_violation_rate
# description: Yes-rate for targets NOT ending in the final digit that all (>=2) examples share (e.g. 84,94,34 -> 40..49); observed < null means the model over-produces yes for final-digit violators (it lets parity/contained-digit/span overlap compensate where people treat the shared final digit as a hard rule), observed > null the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    m = np.zeros(len(t), bool)
    for s in pd.unique(sets):
        xs = [int(x) for x in s.split(",")]
        if len(xs) < 2:
            continue
        fd = {x % 10 for x in xs}
        if len(fd) != 1:
            continue
        d = fd.pop()
        idx = sets == s
        m |= idx & (t % 10 != d)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
