# name: clean_digit_rule_vs_range
# description: In 3-4 example sets whose examples all share a ones digit, yes rate to same-ones-digit targets outside the examples' min-max hull minus yes rate to other-digit targets inside the hull; observed above the null means people let the digit rule dominate the range more than the model, below means the model over-weights the rule relative to the range.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    lo, hi, dg = {}, {}, {}
    for s in sets.unique():
        e = [int(x) for x in s.split(",")]
        lo[s], hi[s], dg[s] = min(e), max(e), -1
        if len(e) >= 3 and len(set(x % 10 for x in e)) == 1:
            dg[s] = e[0] % 10
    L = sets.map(lo).to_numpy()
    H = sets.map(hi).to_numpy()
    m = sets.map(dg).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    clean = m >= 0
    inside = (t > L) & (t < H)
    outside = (t < L) | (t > H)
    a = clean & outside & (t % 10 == m)
    b = clean & inside & (t % 10 != m)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
