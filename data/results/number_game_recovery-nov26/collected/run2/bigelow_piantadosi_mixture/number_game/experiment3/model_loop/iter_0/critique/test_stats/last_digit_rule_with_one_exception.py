# name: last_digit_rule_with_one_exception
# description: Yes rate for non-example targets whose final digit matches exactly 3 of 4 examples (a final-digit rule with one exception, e.g. 71,61,26,81 -> 11); observed above null means people tolerate an exception to a rule more than the model's 10% stray rate allows.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mx = x.max(axis=1)
    mn = x.min(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    k = x.mod(10).eq(t % 10, axis=0).sum(axis=1)
    m = (n == 4) & (k == 3) & ~isin
    v = r[m].mean()
    return float(0.0 if np.isnan(v) else v)
