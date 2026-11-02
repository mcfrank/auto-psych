# name: double_half_target_yes_rate
# description: Yes rate on non-example targets equal to double or half of some example; observed above null means people generalize via multiplicative neighbours more than the model predicts, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    hit = ((ex * 2 == t[:, None]) | (ex == 2 * t[:, None])).any(axis=1)
    isex = (ex == t[:, None]).any(axis=1)
    m = hit & ~isex
    return float(y[m].mean()) if m.sum() > 0 else 0.0
