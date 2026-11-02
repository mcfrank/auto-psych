# name: shared_digit_any_position_small_sets
# description: For sets of 2-3 examples, yes-rate on non-example targets that contain (in any position) a digit present in every example, excluding targets that end in a last digit common to all examples; observed > null means people generalise by a shared digit (e.g. 3,31 -> 38; 42,62 -> 28; 68,58 -> 84) far more than the model, whose "contains digit" family gets near-zero prior mass.
def test_statistic(df):
    t = df["target"].astype(int)
    sets = df["number_set"].astype(str).str.split(",").apply(lambda l: [int(x) for x in l])
    k = sets.str.len()
    common = sets.apply(lambda l: set.intersection(*[set(str(x)) for x in l]))
    lastd = sets.apply(lambda l: {x % 10 for x in l})
    ts = t.astype(str)
    shares = [len(c & set(s)) > 0 for c, s in zip(common, ts)]
    endmatch = [len(c) == 1 and (x % 10) in c for c, x in zip(lastd, t)]
    inset = [x in l for x, l in zip(t, sets)]
    m = (k >= 2) & (k <= 3) & pd.Series(shares, index=df.index) & ~pd.Series(endmatch, index=df.index) & ~pd.Series(inset, index=df.index)
    return float(df.loc[m, "rating"].mean()) if m.any() else 0.0
