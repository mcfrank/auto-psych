# name: digit_rule_below_min_yes_rate
# description: Among sets of 3+ examples all sharing one final digit, yes rate for targets with that same final digit lying below the smallest example; observed above null means people extend a final-digit rule downward past the examples more than the model (with its span-restricted runs) predicts, below null means they confine the rule to the examples' stretch more strongly.
def test_statistic(df):
    parts = df["number_set"].str.split(",", expand=True).astype(float)
    n = parts.notna().sum(axis=1).values
    vals = parts.values
    d = np.mod(vals, 10)
    d0 = d[:, 0]
    same = np.all((d == d0[:, None]) | np.isnan(vals), axis=1)
    mn = np.nanmin(vals, axis=1)
    t = df["target"].astype(float).values
    m = (n >= 3) & same & (np.mod(t, 10) == d0) & (t < mn)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
