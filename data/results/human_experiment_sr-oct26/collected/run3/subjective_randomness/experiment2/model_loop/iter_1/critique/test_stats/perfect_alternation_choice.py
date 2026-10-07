# name: perfect_alternation_choice
# description: Proportion of trials (where exactly one sequence of length >= 4 is perfectly alternating, HTHT...) choosing the perfectly alternating sequence; observed < null means the model over-predicts the appeal of maximal alternation (people detect it as a pattern), > null the reverse.
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    alt_a = (~a.str.contains("HH|TT")).to_numpy(); alt_b = (~b.str.contains("HH|TT")).to_numpy()
    n = a.str.len().to_numpy()
    m = (alt_a != alt_b) & (n >= 4)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    return float(np.where(alt_a[m], y, 1 - y).mean())
