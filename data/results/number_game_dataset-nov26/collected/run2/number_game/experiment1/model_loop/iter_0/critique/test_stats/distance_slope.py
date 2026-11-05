# name: distance_slope
# description: OLS slope of yes (rating) on the target's absolute distance to its nearest example (per 10 units); observed more negative than null means humans show steeper similarity-based generalization gradient than the model.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1) / 10.0
    y = df["rating"].astype(float).to_numpy()
    dv = d - d.mean()
    den = (dv ** 2).sum()
    return float((dv * (y - y.mean())).sum() / den) if den > 0 else 0.0
