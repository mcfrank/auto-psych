# name: mirror_symmetry_avoidance
# description: Among trials (length >= 4) where exactly one sequence is a palindrome or a reversed-complement (mirror-symmetric), the proportion choosing the symmetric sequence as more random; observed < null means people penalise symmetry beyond what the model's switch/motif statistics predict, > null the reverse.
def test_statistic(df):
    comp = str.maketrans("HT", "TH")
    def sym(s):
        r = s.str[::-1]
        return ((s == r) | (s == r.str.translate(comp))).to_numpy()
    a = df["sequence_a"]; b = df["sequence_b"]
    n = a.str.len().to_numpy()
    sa, sb = sym(a), sym(b)
    m = (n >= 4) & (sa != sb)
    y = df["chose_left"].to_numpy()
    if m.sum() == 0:
        return 0.5
    return float(np.where(sa, y, 1 - y)[m].mean())
