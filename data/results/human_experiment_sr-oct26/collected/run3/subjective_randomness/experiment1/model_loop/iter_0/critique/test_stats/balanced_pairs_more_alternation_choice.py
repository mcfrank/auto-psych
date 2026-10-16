# name: balanced_pairs_more_alternation_choice
# description: Among pairs whose head counts differ by at most one but different numbers of alternations, the rate of choosing the sequence with more alternations; observed > null means people favour alternation (when balance is nearly tied) more than the model's alternation term predicts.
def test_statistic(df):
    def alts(s):
        return sum(s[i] != s[i-1] for i in range(1, len(s)))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    d = {s: alts(s) for s in u}
    aa = df["sequence_a"].map(d).values; ab = df["sequence_b"].map(d).values
    ha = df["sequence_a"].str.count("H").values; hb = df["sequence_b"].str.count("H").values
    m = (np.abs(ha - hb) <= 1) & (aa != ab)
    if m.sum() == 0:
        return 0.5
    ch = np.where(aa > ab, df["chose_left"].values, 1 - df["chose_left"].values)
    return float(ch[m].mean())
