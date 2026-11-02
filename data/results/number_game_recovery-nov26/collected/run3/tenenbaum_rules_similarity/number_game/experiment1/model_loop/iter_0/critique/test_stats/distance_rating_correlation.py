# name: distance_rating_correlation
# description: Pearson correlation between yes and log(1+distance from target to nearest example), among targets not in the set; more negative observed than model means people show stronger numerical-similarity gradients than the model predicts.
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
    d = np.nanmin(np.abs(X - t[:, None]), axis=1)
    m = d > 0
    a = np.log1p(d[m]); b = r[m]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
