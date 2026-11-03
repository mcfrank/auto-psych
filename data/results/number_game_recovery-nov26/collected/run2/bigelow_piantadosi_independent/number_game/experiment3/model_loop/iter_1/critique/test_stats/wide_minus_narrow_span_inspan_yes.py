# name: wide_minus_narrow_span_inspan_yes
# description: Among non-example targets inside the span of 3-4-example sets, yes-rate for wide-span sets (span >= 50) minus narrow-span sets (span <= 20); observed below null_mean means people fill in wide spans less than the interval model's flat in-span coverage predicts (above = they fill wide spans more).
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    u = df["number_set"].unique()
    lo = df["number_set"].map({s: min(int(x) for x in s.split(",")) for s in u}).to_numpy()
    hi = df["number_set"].map({s: max(int(x) for x in s.split(",")) for s in u}).to_numpy()
    n = df["number_set"].map({s: s.count(",") + 1 for s in u}).to_numpy()
    isex = np.array([str(tt) in s.split(",") for s, tt in zip(df["number_set"].to_numpy(), t)])
    r = df["rating"].astype(float).to_numpy()
    base = (n >= 3) & (t > lo) & (t < hi) & ~isex
    w = base & (hi - lo >= 50); nr = base & (hi - lo <= 20)
    if w.sum() == 0 or nr.sum() == 0:
        return 0.0
    return float(r[w].mean() - r[nr].mean())
