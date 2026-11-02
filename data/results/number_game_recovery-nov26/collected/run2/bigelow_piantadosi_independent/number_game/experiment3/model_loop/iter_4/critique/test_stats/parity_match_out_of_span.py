# name: parity_match_out_of_span
# description: For sets of >=2 examples that are all odd or all even, yes rate for out-of-span targets matching the examples' parity minus mismatching ones; observed above null means parity generalizes beyond the examples' span more than the model predicts (below null: less).
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    exl = df["number_set"].astype(str).str.split(",")
    n = exl.apply(len).to_numpy()
    odd_ct = exl.apply(lambda x: sum(int(v) % 2 for v in x)).to_numpy()
    lo = exl.apply(lambda x: min(int(v) for v in x)).to_numpy()
    hi = exl.apply(lambda x: max(int(v) for v in x)).to_numpy()
    pure = (n >= 2) & ((odd_ct == 0) | (odd_ct == n))
    setpar = (odd_ct == n).astype(int)
    out = (t < lo) | (t > hi)
    match = (t % 2) == setpar
    r = df["rating"].to_numpy().astype(float)
    a, b = pure & out & match, pure & out & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
