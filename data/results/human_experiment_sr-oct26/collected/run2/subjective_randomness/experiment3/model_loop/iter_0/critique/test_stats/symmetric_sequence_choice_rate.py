# name: symmetric_sequence_choice_rate
# description: Among pairs where exactly one sequence is mirror-symmetric (a palindrome or equal to its reversed H/T complement), the rate the symmetric one is chosen; observed below null means people see symmetry as designed/non-random, which the model does not penalise.
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    tr = str.maketrans("HT", "TH")
    sym = {s: float(s == s[::-1] or s == s[::-1].translate(tr)) for s in seqs}
    sa = df["sequence_a"].map(sym).to_numpy(); sb = df["sequence_b"].map(sym).to_numpy()
    m = sa != sb
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where((sa > sb)[m], c, 1 - c)))
