# name: perfect_alternation_choice_rate
# description: Among trials where exactly one sequence strictly alternates (HTHT.../THTH...), the proportion choosing that sequence; observed above the null means people favour perfect alternation more than the quadratic ideal-switch-rate term predicts, below means they reject it as too regular more than predicted.
def test_statistic(df):
    def alt(s):
        return all(x != y for x, y in zip(s, s[1:]))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: alt(s) for s in u}
    a = df["sequence_a"].map(m).astype(bool).to_numpy()
    b = df["sequence_b"].map(m).astype(bool).to_numpy()
    sel = a ^ b
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(a, c, 1 - c)
    return float(chose[sel].mean())
