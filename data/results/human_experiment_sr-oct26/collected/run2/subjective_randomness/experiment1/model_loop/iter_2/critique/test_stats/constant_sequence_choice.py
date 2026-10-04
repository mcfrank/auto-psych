# name: constant_sequence_choice
# description: Rate of choosing an all-H or all-T sequence when paired with a mixed one; observed below null_mean means the model under-penalises constant sequences.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ca = (a.str.count("(?=HT|TH)") == 0).to_numpy(); cb = (b.str.count("(?=HT|TH)") == 0).to_numpy()
    m = ca ^ cb
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy(dtype=float)[m]
    return float(np.mean(np.where(ca[m], y, 1 - y)))
