# name: perfect_alternation_choice
# description: Among pairs of length >= 4 where exactly one sequence perfectly alternates (HTHT.../THTH...), the rate of choosing the perfectly alternating sequence; observed above null means the model over-penalises perfect alternation (people find it more random than predicted), below means it under-penalises it.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    n = a.str.len()
    pa = ((a.str.count("HT") + a.str.count("TH")) == n - 1).values
    pb = ((b.str.count("HT") + b.str.count("TH")) == n - 1).values
    m = (n.values >= 4) & (pa != pb)
    if m.sum() == 0:
        return 0.5
    chose_alt = np.where(pa, df["chose_left"].values, 1 - df["chose_left"].values)
    return float(chose_alt[m].mean())
