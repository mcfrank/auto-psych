# name: single_example_near_minus_far
# description: On one-example sets, yes rate for non-example targets within 10 of the example minus yes rate for targets more than 40 away; observed below null means people show less magnitude-proximity generalization from a single example than the model's interval concepts predict (observed is roughly flat/negative).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    U = E % 10; tu = t % 10
    def rate(m):
        return float(y[m].mean()) if m.any() else 0.0
    d = np.abs(t - E[:, 0])
    base = (n == 1) & ~is_ex
    return rate(base & (d <= 10)) - rate(base & (d > 40))
