# name: distinct_triplet_choice_rate
# description: Among pairs whose sequences differ in the number of distinct length-3 substrings (motif variety), the proportion of choices of the sequence with more distinct triplets; observed > null means people favour pattern variety / incompressibility beyond what alternation, balance and longest-run terms produce.
def test_statistic(df):
    def ntrip(s):
        s = str(s)
        return len({s[i:i+3] for i in range(len(s) - 2)})
    ua = df["sequence_a"].map({u: ntrip(u) for u in df["sequence_a"].unique()})
    ub = df["sequence_b"].map({u: ntrip(u) for u in df["sequence_b"].unique()})
    m = (ua != ub).values
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].values[m]
    left_more = (ua > ub).values[m]
    return float(np.mean(np.where(left_more, c, 1 - c)))
