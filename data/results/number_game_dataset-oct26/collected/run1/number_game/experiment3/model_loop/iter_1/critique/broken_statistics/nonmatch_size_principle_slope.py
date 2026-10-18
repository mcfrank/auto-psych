# name: nonmatch_size_principle_slope
# description: OLS slope of yes-rate on number of examples (1-4) among targets whose final digit matches no example and whose parity is opposite to every example; observed more negative than null means people narrow generalisation with more examples (size principle) more than the model does, more positive means the reverse.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",")
    n = parts.str.len().astype(float)
    t = df["target"].astype(int)
    ex = df["number_set"].astype(str)
    uniq = ex.unique()
    dmap = {s: set(int(v) % 10 for v in s.split(",")) for s in uniq}
    pmap = {s: set(int(v) % 2 for v in s.split(",")) for s in uniq}
    key = pd.DataFrame({"s": ex, "t": t})
    pairs = key.drop_duplicates()
    good = {(a, b): ((b % 10) not in dmap[a]) and ((b % 2) not in pmap[a]) for a, b in zip(pairs["s"], pairs["t"])}
    sel = pd.Series([good[k] for k in zip(ex, t)], index=df.index, dtype=bool)
    y = df["rating"].astype(float)[sel].values
    xv = n[sel].values
    if len(y) < 3 or np.var(xv) == 0:
        return 0.0
    return float(np.cov(xv, y, bias=True)[0, 1] / np.var(xv))
