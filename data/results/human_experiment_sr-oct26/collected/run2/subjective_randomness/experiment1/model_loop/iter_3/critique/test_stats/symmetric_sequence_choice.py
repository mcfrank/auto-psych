# name: symmetric_sequence_choice
# description: Among pairs where exactly one sequence is mirror-symmetric (a palindrome or the reverse of its H/T complement, e.g. HTTHHTTH, HHHHTTTT) and the other is not, rate of choosing the symmetric one; observed below null_mean means people penalise visible symmetry that the switch-rate+periodicity model does not capture.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    u = pd.unique(pd.concat([a, b]))
    tr = str.maketrans("HT", "TH")
    sym = {s: float(s == s[::-1] or s == s[::-1].translate(tr)) for s in u}
    xa = a.map(sym).to_numpy(); xb = b.map(sym).to_numpy()
    m = xa != xb
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy(dtype=float)[m]
    return float(np.mean(np.where((xa > xb)[m], y, 1 - y)))
