# name: constant_sequence_choice_rate
# description: Among trials where exactly one sequence is a single repeated outcome (all H or all T), the proportion choosing that constant sequence; observed above the null means people choose the all-same sequence more often than the model predicts (e.g. it over-penalises it against other degenerate sequences), below means less often.
def test_statistic(df):
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: len(set(s)) == 1 for s in u}
    a = df["sequence_a"].map(m).astype(bool).to_numpy()
    b = df["sequence_b"].map(m).astype(bool).to_numpy()
    sel = a ^ b
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(a, c, 1 - c)
    return float(chose[sel].mean())
