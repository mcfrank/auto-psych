# name: proximity_gradient
# description: Among targets not in the set, yes rate when the nearest example is 1-5 away minus yes rate when it is 16-30 away; observed > model means the model's numerical-similarity generalization is too flat (people generalize more sharply to nearby numbers), observed < model the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    tg = df["target"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    d = np.where(valid, np.abs(ex - tg[:, None]), np.inf).min(axis=1)
    r = df["rating"].astype(float).to_numpy()
    near = (d >= 1) & (d <= 5)
    far = (d >= 16) & (d <= 30)
    if not near.any() or not far.any():
        return 0.0
    return float(r[near].mean() - r[far].mean())
