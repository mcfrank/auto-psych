# name: unsupported_round_inside_vs_outside_range
# description: Among sets with no multiple of 10, yes rate for round targets inside the examples' [min,max] range minus round targets outside it; observed above null means the round-number doubt is weaker (or confined) within the examples' span than the model's flat veto implies, below null means it is stronger inside the range.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(int).values
    has_round = (np.nan_to_num(ex, nan=1) % 10 == 0).any(1)
    lo = np.nanmin(ex, 1); hi = np.nanmax(ex, 1)
    r = df["rating"].astype(float).values
    m = ~has_round & (t % 10 == 0)
    inside = (t >= lo) & (t <= hi)
    a = r[m & inside]; b = r[m & ~inside]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
