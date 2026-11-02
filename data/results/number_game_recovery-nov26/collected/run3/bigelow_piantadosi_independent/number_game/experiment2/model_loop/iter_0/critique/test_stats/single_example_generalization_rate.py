# name: single_example_generalization_rate
# description: Yes rate on non-example targets for sets with exactly one example; observed above null means the model generalizes too narrowly from a single number (the size principle and commitment are too strong at n=1), and below null means it generalizes too broadly.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    mind = ex.sub(t, axis=0).abs().min(axis=1)
    n = ex.notna().sum(axis=1)
    m = ((n == 1) & (mind > 0)).values
    return float(r[m].mean()) if m.sum() > 0 else 0.0
