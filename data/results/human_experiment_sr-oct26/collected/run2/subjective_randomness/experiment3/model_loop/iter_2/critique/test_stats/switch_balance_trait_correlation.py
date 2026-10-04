# name: switch_balance_trait_correlation
# description: Across participants, the correlation between their rate of choosing the more-switching sequence (pairs differing in switches) and their rate of choosing the more count-balanced sequence (pairs differing in |#H-#T|); observed above null means individual preferences are more positively coupled (a common "randomness-sense" trait) than the model's independent person effects produce, below means more opposed.
def test_statistic(df):
    def sw(col):
        s = df[col]
        return sum((s.str[i] != s.str[i+1]).astype(int).where(s.str.len() > i+1, 0) for i in range(7)).to_numpy()
    sa, sb = sw("sequence_a"), sw("sequence_b")
    ia = (2 * df["sequence_a"].str.count("H") - df["sequence_a"].str.len()).abs().to_numpy()
    ib = (2 * df["sequence_b"].str.count("H") - df["sequence_b"].str.len()).abs().to_numpy()
    c = df["chose_left"].to_numpy(); pid = df["participant_id"].to_numpy()
    m1 = sa != sb
    x1 = pd.Series(np.where(sa[m1] > sb[m1], c[m1], 1 - c[m1])).groupby(pid[m1]).mean()
    m2 = ia != ib
    x2 = pd.Series(np.where(ia[m2] < ib[m2], c[m2], 1 - c[m2])).groupby(pid[m2]).mean()
    j = pd.concat([x1, x2], axis=1, join="inner").dropna()
    if len(j) < 3 or j.iloc[:, 0].std() == 0 or j.iloc[:, 1].std() == 0:
        return 0.0
    return float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])
