# name: perfect_alternation_choice
# description: Rate of choosing a perfectly alternating sequence (HTHT..., length>=4) when paired with a non-alternating one; observed below null_mean means the model over-predicts the appeal of perfect alternation.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    la = a.str.len(); lb = b.str.len()
    aa = ((a.str.count("(?=HT|TH)") == la - 1) & (la >= 4)).to_numpy()
    ab = ((b.str.count("(?=HT|TH)") == lb - 1) & (lb >= 4)).to_numpy()
    m = aa ^ ab
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy(dtype=float)[m]
    return float(np.mean(np.where(aa[m], y, 1 - y)))
