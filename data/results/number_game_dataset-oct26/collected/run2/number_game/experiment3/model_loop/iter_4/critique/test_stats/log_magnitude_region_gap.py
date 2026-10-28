# name: log_magnitude_region_gap
# description: Among out-of-span targets 5-40 (linear) from the nearest span edge, yes-rate when that nearest edge is <=20 minus when it is >=60; observed < null means equal linear gaps hurt more among small numbers (log-like magnitude compression), which the model's linear size distance under-produces.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int).values
    mn = ex.min(axis=1).values; mx = ex.max(axis=1).values
    below = t < mn; above = t > mx
    gap = np.where(below, mn - t, np.where(above, t - mx, 0))
    edge = np.where(below, mn, mx)
    sel = (below | above) & (gap >= 5) & (gap <= 40)
    r = df["rating"].values.astype(float)
    a = sel & (edge <= 20); b = sel & (edge >= 60)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
