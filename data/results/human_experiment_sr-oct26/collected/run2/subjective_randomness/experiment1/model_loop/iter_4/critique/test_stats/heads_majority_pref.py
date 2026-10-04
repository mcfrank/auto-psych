# name: heads_majority_pref
# description: Among pairs with equal switch counts whose H-counts differ, rate of choosing the sequence with more H; the incumbent is H/T symmetric (null near 0.5), so observed above/below null_mean means people favour heads-heavy/tails-heavy sequences as more random.
def test_statistic(df):
    def sw(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    swm = {s: sw(s) for s in u}
    ha, hb = df["sequence_a"].str.count("H"), df["sequence_b"].str.count("H")
    m = (df["sequence_a"].map(swm) == df["sequence_b"].map(swm)) & (ha != hb)
    if m.sum() == 0:
        return 0.5
    chose_h = np.where(ha[m] > hb[m], df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(chose_h))
