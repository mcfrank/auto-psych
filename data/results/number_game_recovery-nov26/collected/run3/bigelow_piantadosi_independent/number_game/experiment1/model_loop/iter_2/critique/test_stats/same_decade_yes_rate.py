# name: same_decade_yes_rate
# description: Yes rate for non-example targets in the same decade (same tens digit, 1-100 with 100 in its own decade) as at least one example minus yes rate for targets in a decade containing no example; observed > null means people use decade/tens-digit similarity more than the model, observed < null means less.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    vals = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    is_ex = np.any(vals == t[:, None], axis=1)
    dec = np.floor(vals / 10)
    same = np.any(dec == np.floor(t / 10)[:, None], axis=1)
    a = same & ~is_ex
    b = ~same & ~is_ex
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
