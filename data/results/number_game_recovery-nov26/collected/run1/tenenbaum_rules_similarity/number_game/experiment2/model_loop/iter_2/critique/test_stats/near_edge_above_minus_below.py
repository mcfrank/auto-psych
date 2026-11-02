# name: near_edge_above_minus_below
# description: Yes rate for targets 1-10 above the largest example minus yes rate for targets 1-10 below the smallest example (all sets); observed below null means people extend generalisation downward more than upward relative to the model (above null: upward more), i.e. a directional asymmetry the symmetric interval prior lacks.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    a = (t > hi) & (t <= hi + 10)
    b = (t < lo) & (t >= lo - 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
