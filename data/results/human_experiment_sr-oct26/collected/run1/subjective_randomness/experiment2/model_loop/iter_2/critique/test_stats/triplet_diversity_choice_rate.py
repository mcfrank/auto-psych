# name: triplet_diversity_choice_rate
# description: Among trials (length >= 5) where the two sequences contain different numbers of distinct length-3 substrings (e.g. HHT, THT), the proportion choosing the sequence with MORE distinct triplets; observed above the null means people reward local pattern variety beyond what the alternation/balance/streak/symmetry terms predict, below means less.
def test_statistic(df):
    def ntrip(s):
        return len({s[i:i + 3] for i in range(len(s) - 2)})
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: ntrip(s) for s in u}
    ta = df["sequence_a"].map(m).to_numpy(dtype=float)
    tb = df["sequence_b"].map(m).to_numpy(dtype=float)
    L = df["sequence_a"].str.len().to_numpy()
    sel = (ta != tb) & (L >= 5)
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(ta > tb, c, 1 - c)
    return float(chose[sel].mean())
