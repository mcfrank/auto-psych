# name: near_nonmember_yes_rate
# description: Yes-rate for targets 1-5 away from the nearest example (not equal to any example); observed above null means humans generalise to numeric neighbours more than the model's rule/interval posterior predicts, below null means less.
def _ex(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values.astype(float)
    r = df["rating"].astype(float).values
    return ex, t, r

def test_statistic(df):
    ex, t, r = _ex(df)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (d >= 1) & (d <= 5)
    return float(r[m].mean()) if m.any() else 0.0
