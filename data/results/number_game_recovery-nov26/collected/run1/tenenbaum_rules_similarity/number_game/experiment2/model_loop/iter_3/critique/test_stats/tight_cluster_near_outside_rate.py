# name: tight_cluster_near_outside_rate
# description: Yes rate for targets 1-5 outside the example span (below min or above max) on sets of >=2 examples spanning <=10; observed > null means the model rejects targets just outside a tight cluster more than people do (overly strong size principle), observed < null means people are even tighter.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    k = ex.notna().sum(axis=1)
    lo, hi = ex.min(axis=1), ex.max(axis=1)
    t = df["target"].astype(int)
    d = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0))
    m = (k >= 2) & ((hi - lo) <= 10) & (d >= 1) & (d <= 5)
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].mean())
