# name: last_vs_first_example_digit_match_gap
# description: In multi-example sets, among non-example targets sharing a final digit with exactly one example, yes rate when that example was shown last minus when it was shown first; t_observed different from null_mean (~0) means presentation order matters (above = recency, below = primacy), which the order-free model cannot produce.
def test_statistic(df):
    t = df["target"].astype(int).values
    sets = df["number_set"].astype(str).values
    pos = np.full(len(t), -1)
    for s in np.unique(sets):
        xs = np.array([int(v) for v in s.split(",")])
        if len(xs) < 2:
            continue
        m = sets == s
        tt = t[m]
        match = (tt[:, None] % 10) == (xs[None, :] % 10)
        isex = (tt[:, None] == xs[None, :]).any(1)
        one = (match.sum(1) == 1) & ~isex
        idx = match.argmax(1)
        p = np.where(one, np.where(idx == len(xs) - 1, 1, np.where(idx == 0, 0, -1)), -1)
        pos[m] = p
    r = df["rating"].astype(float).values
    last = pos == 1
    first = pos == 0
    if last.sum() == 0 or first.sum() == 0:
        return 0.0
    return float(r[last].mean() - r[first].mean())
