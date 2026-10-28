# name: small_vs_large_region_gap
# description: Among targets outside the span by a linear gap of 5-40, yes-rate when the nearest example is <=20 minus when the nearest example is >=60; observed < null means people compress large numbers (log-like magnitude) so equal linear gaps matter more among small numbers, which the linear-distance model fails to produce.
def test_statistic(df):
    g = df.groupby(["number_set", "target"], sort=False)["rating"].mean().reset_index()
    small, large = [], []
    for s, t, r in zip(g["number_set"], g["target"].astype(int), g["rating"]):
        ex = [int(x) for x in str(s).split(",")]
        gap = max(min(ex) - t, t - max(ex), 0)
        if gap < 5 or gap > 40:
            continue
        x = min(ex, key=lambda e: abs(e - t))
        if x <= 20:
            small.append(r)
        elif x >= 60:
            large.append(r)
    if not small or not large:
        return 0.0
    return float(np.mean(small) - np.mean(large))
