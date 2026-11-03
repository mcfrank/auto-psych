# name: span_center_vs_edge_gap
# description: On 3-4 example sets with span at least 20, among non-example targets strictly inside the span, yes rate for targets in the middle 40% of the span minus those in the outer 20% at each end; observed above null means people treat the span centre as more typical than the model's flat interval hypotheses, below null means edges near examples are favoured.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    n = (~np.isnan(ex)).sum(1)
    mx = np.nanmax(ex, 1); mn = np.nanmin(ex, 1)
    isex = (ex == t[:, None]).any(1)
    span = mx - mn
    pos = (t - mn) / np.where(span > 0, span, 1)
    m = (n >= 3) & (span >= 20) & ~isex & (t > mn) & (t < mx)
    a = r[m & (pos >= 0.3) & (pos <= 0.7)]; b = r[m & ((pos < 0.2) | (pos > 0.8))]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
