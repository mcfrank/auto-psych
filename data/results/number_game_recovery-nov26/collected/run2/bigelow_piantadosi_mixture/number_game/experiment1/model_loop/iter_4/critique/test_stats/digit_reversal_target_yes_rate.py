# name: digit_reversal_target_yes_rate
# description: Yes rate on two-digit targets whose digit reversal (e.g. 21 for 12) is one of the examples, excluding targets that are themselves examples; observed above null means people use a digit-reversal similarity the model under-produces.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    rev = (t % 10) * 10 + (t // 10)
    two = (t >= 10) & (t <= 99) & (t % 10 != 0) & (rev != t)
    hit = (ex == rev[:, None]).any(axis=1)
    isex = (ex == t[:, None]).any(axis=1)
    m = two & hit & ~isex
    return float(y[m].mean()) if m.sum() > 0 else 0.0
