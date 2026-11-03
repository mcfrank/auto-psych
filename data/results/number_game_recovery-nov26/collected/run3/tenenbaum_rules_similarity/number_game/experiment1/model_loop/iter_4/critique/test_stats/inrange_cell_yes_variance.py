# name: inrange_cell_yes_variance
# description: Variance across (set,target) cells of the cell "yes" rate, among cells whose target lies inside the examples' range in sets with >=3 examples; observed above null = people's in-range judgements are more all-or-none (more polarised by target) than the model's graded predictions.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    t = df["target"].astype(int)
    lo, hi = ex.min(axis=1), ex.max(axis=1)
    m = (n >= 3) & (t > lo) & (t < hi)
    sub = df.loc[m]
    if len(sub) == 0:
        return 0.0
    cm = sub.groupby(["number_set", "target"])["rating"].mean()
    return float(cm.var(ddof=0))
