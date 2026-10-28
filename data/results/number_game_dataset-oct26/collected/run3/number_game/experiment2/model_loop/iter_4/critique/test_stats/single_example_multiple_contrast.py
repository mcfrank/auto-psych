# name: single_example_multiple_contrast
# description: Among one-example sets with example >= 3, yes-rate for targets that are multiples or divisors (>1) of the example minus other targets; observed above null means the model under-produces multiplicative generalisation from one example.
def test_statistic(df):
    s = df["number_set"].astype(str)
    m = (s.str.count(",") == 0).values
    ex = pd.to_numeric(s[m], errors="coerce").fillna(1).astype(int).values
    t = df["target"].astype(int).values[m]
    r = df["rating"].astype(float).values[m]
    keep = ex >= 3
    ex, t, r = ex[keep], t[keep], r[keep]
    if len(t) == 0:
        return 0.0
    rel = ((t % ex) == 0) | (((ex % np.maximum(t, 1)) == 0) & (t > 1))
    a = r[rel].mean() if rel.any() else 0.0
    b = r[~rel].mean() if (~rel).any() else 0.0
    return float(a - b)
