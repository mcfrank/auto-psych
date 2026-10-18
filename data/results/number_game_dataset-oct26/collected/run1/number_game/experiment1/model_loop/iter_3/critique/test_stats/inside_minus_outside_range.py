# name: inside_minus_outside_range
# description: On 2-4 example sets, yes rate for targets strictly inside the examples' min-max range minus targets outside it; observed above null means people use interval/range-based generalization that the magnitude-blind digit model lacks, below null means they favour out-of-range targets more than predicted.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",")
    ex = pd.DataFrame(sets.tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    r = df["rating"].to_numpy().astype(float)
    inside = (n >= 2) & (t > lo) & (t < hi)
    outside = (n >= 2) & ((t < lo) | (t > hi))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
