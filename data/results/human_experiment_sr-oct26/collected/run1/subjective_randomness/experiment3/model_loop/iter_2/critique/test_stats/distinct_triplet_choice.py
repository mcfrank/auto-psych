# name: distinct_triplet_choice
# description: Among length >= 7 trials whose sequences differ in the number of DISTINCT length-3 substrings (local pattern variety), the proportion choosing the sequence with more distinct triplets; observed above null_mean means people reward local pattern complexity beyond what the model's switch/run/symmetry features capture, below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    def tri(s):
        u = pd.unique(s)
        return s.map({x: len({x[i:i + 3] for i in range(len(x) - 2)}) for x in u}).to_numpy()
    ta, tb = tri(a), tri(b)
    mask = (a.str.len().to_numpy() >= 7) & (ta != tb)
    if not mask.any():
        return 0.5
    c = df["chose_left"].to_numpy()
    chose_more = np.where(ta > tb, c, 1 - c)
    return float(chose_more[mask].mean())
