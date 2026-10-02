# name: short_length_alternation_preference
# description: Proportion choosing the sequence with more alternations among pairs of length <= 7 minus the same among length-8 pairs (unequal alternation counts only); observed != null means sensitivity to alternation depends on sequence length in a way the model's length-normalised rate terms miss (observed < null: weaker preference on short sequences than predicted).
def test_statistic(df):
    def nalt(s):
        s = str(s)
        return sum(1 for x, y in zip(s, s[1:]) if x != y)
    ua = df["sequence_a"].map({u: nalt(u) for u in df["sequence_a"].unique()}).values
    ub = df["sequence_b"].map({u: nalt(u) for u in df["sequence_b"].unique()}).values
    L = df["sequence_a"].str.len().values
    m = ua != ub
    c = df["chose_left"].values
    chose_more = np.where(ua > ub, c, 1 - c)
    s = m & (L <= 7); l = m & (L == 8)
    if s.sum() == 0 or l.sum() == 0:
        return 0.0
    return float(chose_more[s].mean() - chose_more[l].mean())
