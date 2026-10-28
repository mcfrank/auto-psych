# name: outside_hull_near_yes
# description: Yes rate for targets lying outside [min, max] of a multi-example set but within 10 of that range (and not in the set); observed above null_mean means people extend similarity-based generalization beyond the examples' range more than the model, below means less.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1)
    lo = s.min(axis=1); hi = s.max(axis=1)
    t = df["target"].astype(int)
    gap = np.maximum(lo - t, t - hi)
    m = (n >= 2) & (gap > 0) & (gap <= 10)
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].mean())
