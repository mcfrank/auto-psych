# name: double_or_half_of_example_gap
# description: On 2+ example sets, yes rate for non-example targets equal to twice or half of some example minus other non-example targets; observed above null means people use doubling/halving relations to examples more than the rule-based model predicts, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    n = (~np.isnan(ex)).sum(1)
    mx = np.nanmax(ex, 1); mn = np.nanmin(ex, 1)
    isex = (ex == t[:, None]).any(1)
    hit = ((2 * ex == t[:, None]) | (ex == 2 * t[:, None])).any(1)
    m = (n >= 2) & ~isex
    a = r[m & hit]; b = r[m & ~hit]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
