# name: exception_last_digit_rule_yes
# description: For 4-example sets where exactly 3 examples share a last digit and the 4th does not (e.g. 92,82,27,42), yes rate on non-example targets ending in that shared digit; observed > null means people discount a single exception and keep the "ends in d" rule more than the model's fixed 0.9 example-reliability allows, observed < null means the exception breaks the rule for them more.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    Dg = np.where(valid, np.mod(np.nan_to_num(E), 10), -1)
    cnt = np.stack([(Dg == k).sum(axis=1) for k in range(10)], axis=1)
    maj = np.argmax(cnt, axis=1)
    top = cnt.max(axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    m = (n == 4) & (top == 3) & ~in_set & (np.mod(t, 10) == maj)
    return float(r[m].mean()) if m.sum() else 0.0
