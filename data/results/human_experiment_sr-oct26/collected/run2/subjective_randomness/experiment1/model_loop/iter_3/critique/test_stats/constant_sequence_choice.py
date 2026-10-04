# name: constant_sequence_choice
# description: Among pairs where exactly one sequence is constant (all H or all T), rate of choosing the constant sequence; observed above null_mean means the model over-penalises all-same sequences relative to people (below: under-penalises them).
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ca = (a.str.count("H") == 0) | (a.str.count("T") == 0)
    cb = (b.str.count("H") == 0) | (b.str.count("T") == 0)
    m = (ca != cb).to_numpy()
    if m.sum() == 0:
        return 0.0
    y = df["chose_left"].to_numpy(dtype=float)[m]
    return float(np.mean(np.where(ca.to_numpy()[m], y, 1 - y)))
