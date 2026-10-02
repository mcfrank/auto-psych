# name: perfect_alternation_choice_rate
# description: Among pairs where exactly one sequence is perfectly alternating (HTHT.../THTH...), the proportion of choices of that perfectly alternating sequence; observed < null means people reject perfect alternation as too regular more than the model's smooth ideal-alternation curve predicts (observed > null the reverse).
def test_statistic(df):
    def isalt(s):
        return (s.str.contains("HH") | s.str.contains("TT")) == False
    a = isalt(df["sequence_a"]) & (df["sequence_a"].str.len() >= 3)
    b = isalt(df["sequence_b"]) & (df["sequence_b"].str.len() >= 3)
    m = a != b
    if m.sum() == 0:
        return 0.5
    chose_alt = np.where(a[m], df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(chose_alt))
