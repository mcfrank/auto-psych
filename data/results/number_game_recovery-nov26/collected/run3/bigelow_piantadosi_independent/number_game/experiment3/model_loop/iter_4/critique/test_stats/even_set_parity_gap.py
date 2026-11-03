# name: even_set_parity_gap
# description: On 2+ example sets whose examples are all even, yes rate for even non-example targets minus odd non-example targets; observed above null means people generalise 'even' more sharply than the model, below null means they ignore shared evenness more than predicted.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    n = (~np.isnan(ex)).sum(1)
    mx = np.nanmax(ex, 1); mn = np.nanmin(ex, 1)
    isex = (ex == t[:, None]).any(1)
    e = np.nan_to_num(ex, nan=2)
    alleven = (n >= 2) & (e % 2 == 0).all(1)
    m = alleven & ~isex
    a = r[m & (t % 2 == 0)]; b = r[m & (t % 2 == 1)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
