# name: unstructured_set_nonexample_yes
# description: Yes rate to non-example targets for multi-example sets with mixed parity, no shared units digit and span of at least 40 (sets fitting no familiar rule or narrow range); observed above null means people generalise more broadly from patternless sets than the model, below null means they are more restrictive.
def test_statistic(df):
    s = df["number_set"].astype(str)
    uniq = pd.Series(s.unique())
    def props(x):
        v = [int(a) for a in x.split(",")]
        par = len(set(a % 2 for a in v))
        units = len(set(a % 10 for a in v))
        return len(v) > 1 and par == 2 and units == len(v) and (max(v) - min(v)) >= 40
    flag = dict(zip(uniq, [props(x) for x in uniq]))
    m = s.map(flag).astype(bool)
    t = df["target"].astype(int).astype(str)
    is_ex = (("," + s + ",").str.contains(",", regex=False)) & pd.Series([("," + a + ",").find("," + b + ",") >= 0 for a, b in zip(s.values, t.values)], index=df.index)
    sel = m & ~is_ex
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[sel].mean())
