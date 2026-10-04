# name: palindrome_chosen
# description: Among pairs (length >= 4) where exactly one sequence is a mirror palindrome (reads the same backwards, e.g. HTTH, HHTTHH), the rate the palindrome is chosen; observed below null means the model misses a penalty on mirror-symmetric sequences, above means it over-penalises them.
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    n = a.str.len().to_numpy()
    pa = (a == a.str[::-1]).to_numpy() & (n >= 4)
    pb = (b == b.str[::-1]).to_numpy() & (n >= 4)
    m = pa != pb
    if not m.any():
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where(pa[m], c, 1 - c)))
