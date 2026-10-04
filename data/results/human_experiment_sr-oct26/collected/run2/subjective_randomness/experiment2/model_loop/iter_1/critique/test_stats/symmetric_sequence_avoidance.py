# name: symmetric_sequence_avoidance
# description: Among pairs where exactly one sequence is mirror-symmetric (a palindrome or its H/T-complemented reversal), the rate of choosing the asymmetric one; observed above null means people see symmetry as non-random beyond what the model's periodicity penalty captures (below null: the model over-penalises).
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    u = pd.unique(np.concatenate([a, b]))
    tr = str.maketrans("HT", "TH")
    sym = {s: float(s == s[::-1] or s == s[::-1].translate(tr)) for s in u}
    sa = df["sequence_a"].map(sym).to_numpy(); sb = df["sequence_b"].map(sym).to_numpy()
    m = sa != sb
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.where(sa[m] == 0, c, 1 - c).mean())
