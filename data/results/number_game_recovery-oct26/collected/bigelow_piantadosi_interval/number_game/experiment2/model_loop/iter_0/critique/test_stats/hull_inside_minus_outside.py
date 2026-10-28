# name: hull_inside_minus_outside
# description: Among non-example targets of sets with >= 2 examples, yes rate for targets strictly inside the examples' [min, max] range minus yes rate for targets outside it; observed > null means people generalize by magnitude range (interpolation) more sharply than the model, observed < null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    tg = df["target"].astype(int).values.astype(float)
    k = (~np.isnan(ex)).sum(axis=1)
    mn, mx = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    isex = (ex == tg[:, None]).any(axis=1)
    r = df["rating"].values.astype(float)
    base = (k >= 2) & ~isex
    inside = base & (tg > mn) & (tg < mx)
    outside = base & ((tg < mn) | (tg > mx))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
