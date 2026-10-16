# name: perfect_alternation_choice_rate
# description: For pairs of length >= 5 where exactly one sequence is a perfect alternation (HTHT.. / THTH..), the rate of choosing that alternating sequence as more random; observed above null_mean means the model over-penalises perfect alternation (below: under-penalises).
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    L = a.str.len()
    alt_a = ~a.str.contains("HH|TT", regex=True)
    alt_b = ~b.str.contains("HH|TT", regex=True)
    m = ((alt_a ^ alt_b) & (L >= 5)).values
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].values
    return float(np.mean(np.where(alt_a.values, y, 1 - y)[m]))
