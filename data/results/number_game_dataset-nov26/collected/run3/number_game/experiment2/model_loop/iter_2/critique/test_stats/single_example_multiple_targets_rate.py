# name: single_example_multiple_targets_rate
# description: Yes rate on one-example sets (example > 1) for non-example targets that are multiples of the example (6->18, 11->55, 25->100); observed above null means the neglect model still under-generalizes a lone example to its multiples rule (below = over-generalizes).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    E = pd.DataFrame(ex.tolist()).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    isex = (E == t[:, None]).any(axis=1)
    mn = np.nanmin(E, axis=1); mx = np.nanmax(E, axis=1)
    x = E[:, 0]
    m = (n == 1) & (~isex) & (x > 1) & (np.mod(t, np.where(x > 0, x, 1)) == 0)
    return float(r[m].mean()) if m.any() else 0.0
