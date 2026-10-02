# name: palindrome_choice_rate
# description: Among trials where exactly one sequence is a palindrome (mirror-symmetric, e.g. HTTTTTTH, HHHTTHHH), the proportion choosing the palindrome; observed below the null means people penalise visible symmetry beyond what the model's alternation/balance/streak terms predict, above means they favour it.
def test_statistic(df):
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: (s == s[::-1]) for s in u}
    a = df["sequence_a"].map(m).astype(bool).to_numpy()
    b = df["sequence_b"].map(m).astype(bool).to_numpy()
    sel = a ^ b
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(a, c, 1 - c)
    return float(chose[sel].mean())
