# name: symmetric_sequence_choice_rate
# description: Among pairs (length>=5) where exactly one sequence is mirror-symmetric (a palindrome, or the reverse of its H/T complement, e.g. HTTHHTTH, HHHHTTTT) and neither constant nor perfectly alternating, the proportion of choices for the symmetric sequence; observed below null_mean means people penalise symmetry more than the model predicts, above means the model over-penalises it.
def test_statistic(df):
    def sym(s):
        if len(s) < 5 or len(set(s)) == 1 or all(x != y for x, y in zip(s, s[1:])):
            return False
        comp = s.translate(str.maketrans("HT", "TH"))
        return s == s[::-1] or comp == s[::-1]
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: sym(s) for s in u}
    pa = df["sequence_a"].map(m).astype(bool)
    pb = df["sequence_b"].map(m).astype(bool)
    sel = pa ^ pb
    if sel.sum() == 0:
        return 0.5
    c = np.where(pa[sel], df["chose_left"][sel], 1 - df["chose_left"][sel])
    return float(np.mean(c))
