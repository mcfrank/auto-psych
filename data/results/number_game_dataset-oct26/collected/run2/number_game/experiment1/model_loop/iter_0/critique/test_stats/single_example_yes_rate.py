# name: single_example_yes_rate
# description: Yes-rate on trials with a single example (excluding target == example); observed above null means humans generalise more broadly from one example than the model's size-principle posterior does.
def _ex(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values.astype(float)
    r = df["rating"].astype(float).values
    return ex, t, r

def test_statistic(df):
    ex, t, r = _ex(df)
    n = np.sum(~np.isnan(ex), axis=1)
    m = (n == 1) & (ex[:, 0] != t)
    return float(r[m].mean()) if m.any() else 0.0
