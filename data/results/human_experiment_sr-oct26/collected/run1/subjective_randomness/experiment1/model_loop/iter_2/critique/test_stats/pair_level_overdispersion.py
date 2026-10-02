# name: pair_level_overdispersion
# description: Variance across stimulus pairs (orientation-normalised to the lexicographically smaller sequence) of the proportion of participants choosing that sequence; observed above the null means pairs are judged more decisively (consensus more extreme) than the model predicts, i.e. it under-predicts how strongly some pairs are preferred.
def test_statistic(df):
    a = df["sequence_a"].to_numpy()
    b = df["sequence_b"].to_numpy()
    y = df["chose_left"].to_numpy(float)
    first = np.where(a <= b, a, b)
    second = np.where(a <= b, b, a)
    chose_first = np.where(a <= b, y, 1 - y)
    key = pd.Series(first + "|" + second)
    props = pd.Series(chose_first).groupby(key.to_numpy()).mean()
    return float(props.var(ddof=0))
