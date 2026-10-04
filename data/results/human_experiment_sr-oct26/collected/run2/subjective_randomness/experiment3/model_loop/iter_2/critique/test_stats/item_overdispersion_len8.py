# name: item_overdispersion_len8
# description: Variance across distinct length-8 pairs of the pair's choice rate (oriented to the lexicographically smaller sequence); observed above null means items are more extreme/polarised than the model predicts (it under-produces item-level consensus), below means it over-produces it.
def test_statistic(df):
    d = df[df["sequence_a"].str.len() == 8]
    a = d["sequence_a"]; b = d["sequence_b"]
    first = a < b
    key = np.where(first, a + "|" + b, b + "|" + a)
    y = np.where(first, d["chose_left"], 1 - d["chose_left"])
    r = pd.Series(y).groupby(key).mean()
    return float(r.var()) if len(r) > 1 else 0.0
