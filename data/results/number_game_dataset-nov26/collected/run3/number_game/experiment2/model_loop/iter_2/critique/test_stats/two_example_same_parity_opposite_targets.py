# name: two_example_same_parity_opposite_targets
# description: For two-example sets whose examples share parity, yes rate for non-example targets of the opposite parity; observed above null means people generalize across parity more than the model from two examples (below = people restrict to the shared parity more than the model).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    E = pd.DataFrame(ex.tolist()).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    isex = (E == t[:, None]).any(axis=1)
    mn = np.nanmin(E, axis=1); mx = np.nanmax(E, axis=1)
    p0 = np.mod(E[:, 0], 2); p1 = np.mod(E[:, 1], 2) if E.shape[1] > 1 else np.full(len(t), np.nan)
    m = (n == 2) & (~isex) & (p0 == p1) & (np.mod(t, 2) != p0)
    return float(r[m].mean()) if m.any() else 0.0
