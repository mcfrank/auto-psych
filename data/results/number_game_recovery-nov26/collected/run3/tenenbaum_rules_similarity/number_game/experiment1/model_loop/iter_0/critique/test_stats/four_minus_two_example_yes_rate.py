# name: four_minus_two_example_yes_rate
# description: Yes rate on 4-example sets minus yes rate on 2-example sets; observed above the model mean means people generalize more (not less) with more examples than the size principle predicts.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")][:4]
        E[i, :len(v)] = v
    X = E[codes]  # (rows, 4) examples, NaN padded
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(X), axis=1)
    a = r[n == 4]; b = r[n == 2]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
