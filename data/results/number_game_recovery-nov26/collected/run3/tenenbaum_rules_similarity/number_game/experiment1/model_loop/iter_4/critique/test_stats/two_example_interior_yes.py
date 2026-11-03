# name: two_example_interior_yes
# description: For two-example sets, the "yes" rate for targets strictly between the two examples that share neither example's last digit; observed below null = model's range concepts over-produce interior "yes" for two-example sets (people stay rule-bound), above = people interpolate more than the model.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    t = df["target"].astype(int)
    lo, hi = ex.min(axis=1), ex.max(axis=1)
    sharedig = ex.mod(10).eq(t % 10, axis=0).any(axis=1)
    m = (n == 2) & (t > lo) & (t < hi) & ~sharedig
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].mean())
