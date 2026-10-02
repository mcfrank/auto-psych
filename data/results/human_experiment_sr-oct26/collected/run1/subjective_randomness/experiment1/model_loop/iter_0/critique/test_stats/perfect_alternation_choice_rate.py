# name: perfect_alternation_choice_rate
# description: On trials where exactly one sequence perfectly alternates (HTHT...), the fraction choosing that perfectly alternating sequence; observed < null means people reject the strict alternating pattern more than the model's periodicity penalty predicts, observed > null means less.
def test_statistic(df):
    def perfalt(s):
        return float(len(s) > 2 and all(s[i] != s[i - 1] for i in range(1, len(s))))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: perfalt(s) for s in u}
    a = df["sequence_a"].map(m).to_numpy()
    b = df["sequence_b"].map(m).to_numpy()
    y = df["chose_left"].to_numpy()
    mask = (a + b) == 1
    if mask.sum() == 0:
        return 0.5
    chose_alt = np.where(a == 1, y, 1 - y)
    return float(chose_alt[mask].mean())
