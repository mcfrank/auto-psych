# name: upward_minus_downward_extension
# description: Yes-rate for targets 1-20 above the set maximum minus yes-rate for targets 1-20 below the set minimum (multi-example sets); the model's intervals are symmetric (up to edge effects), so observed above the null means people extend generalization upward more than downward than the model predicts, below means downward more.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    t = df["target"].astype(int)
    lo, hi = ex.min(axis=1), ex.max(axis=1)
    r = df["rating"].astype(float)
    up = (n >= 2) & (t > hi) & (t <= hi + 20)
    dn = (n >= 2) & (t < lo) & (t >= lo - 20)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
