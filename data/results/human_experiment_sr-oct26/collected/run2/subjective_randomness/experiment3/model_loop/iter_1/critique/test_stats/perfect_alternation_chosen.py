# name: perfect_alternation_chosen
# description: Among pairs where exactly one sequence (length >= 3) perfectly alternates (HTHT..), the rate it is chosen as more random; observed below null means the model over-predicts choosing pure alternations, above means it under-predicts.
def test_statistic(df):
    def alt(s):
        n = s.str.len()
        return (n >= 3) & ~s.str.contains("HH") & ~s.str.contains("TT")
    aa, ab = alt(df["sequence_a"]).to_numpy(), alt(df["sequence_b"]).to_numpy()
    m = aa != ab
    if not m.any():
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where(aa[m], c, 1 - c)))
