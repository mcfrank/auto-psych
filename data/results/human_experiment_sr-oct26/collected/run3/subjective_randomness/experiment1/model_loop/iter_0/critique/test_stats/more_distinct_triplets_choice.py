# name: more_distinct_triplets_choice
# description: Among pairs whose number of distinct length-3 substrings differs, the rate of choosing the sequence with more distinct triplets (motif diversity); observed > null means people reward pattern diversity more than the model predicts.
def test_statistic(df):
    def nd(s):
        return len({s[i:i+3] for i in range(len(s) - 2)})
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    d = {s: nd(s) for s in u}
    ra = df["sequence_a"].map(d).values; rb = df["sequence_b"].map(d).values
    m = ra != rb
    if m.sum() == 0:
        return 0.5
    ch = np.where(ra > rb, df["chose_left"].values, 1 - df["chose_left"].values)
    return float(ch[m].mean())
