# name: arithmetic_next_term_yes_gap
# description: On 2+ example sets, yes rate for the target equal to last shown example plus (last minus second-to-last) minus yes rate for other non-example targets of the same sets; observed above null means people extrapolate the shown sequence's step more than the concept model predicts, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    n = (~np.isnan(ex)).sum(1)
    mx = np.nanmax(ex, 1); mn = np.nanmin(ex, 1)
    isex = (ex == t[:, None]).any(1)
    rows = np.arange(len(t))
    li = np.clip(n - 1, 0, None); pi = np.clip(n - 2, 0, None)
    last = ex[rows, li]; prev = ex[rows, pi]
    nxt = 2 * last - prev
    m = (n >= 2) & ~isex
    hit = m & (t == nxt) & (last != prev)
    a = r[hit]; b = r[m & ~hit]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
