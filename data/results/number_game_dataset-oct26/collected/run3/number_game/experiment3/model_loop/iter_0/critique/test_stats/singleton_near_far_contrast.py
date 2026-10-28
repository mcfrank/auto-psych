# name: singleton_near_far_contrast
# description: For one-example sets, yes-rate for targets within 10 of the example (excluding the example itself) minus yes-rate for targets more than 30 away; observed > null means people generalize from a single example more locally (similarity-based) than the model's rule/interval mixture plus guessing predicts.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    ex = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    E = ex[codes]  # (rows, 4) examples, NaN-padded, in shown order
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    s = n == 1
    d = np.abs(E[:, 0] - t)
    near = s & (d > 0) & (d <= 10)
    far = s & (d > 30)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(y[near].mean() - y[far].mean())
