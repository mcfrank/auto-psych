# name: palindrome_choice
# description: Among length-8 pairs where exactly one sequence is a palindrome (reads the same reversed) and the two differ in switch count by at most 1, the rate of choosing the palindrome; observed below null means mirror symmetry makes a sequence look less random than the model (which has no symmetry detector) predicts, above means more.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    pa = (a == a.str[::-1]).to_numpy(); pb = (b == b.str[::-1]).to_numpy()
    ka = (a.str.count("HT") + a.str.count("TH")).to_numpy()
    kb = (b.str.count("HT") + b.str.count("TH")).to_numpy()
    n = a.str.len().to_numpy()
    m = (n == 8) & (pa != pb) & (np.abs(ka - kb) <= 1)
    y = df["chose_left"].to_numpy()[m]
    ta = pa[m]
    if len(y) == 0:
        return 0.5
    return float(np.mean(np.where(ta, y, 1 - y)))
