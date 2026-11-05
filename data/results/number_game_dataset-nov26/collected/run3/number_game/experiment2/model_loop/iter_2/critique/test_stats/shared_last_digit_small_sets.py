# name: shared_last_digit_small_sets
# description: For sets with 1-2 examples, yes rate for non-example targets sharing a last digit with some example (14->74, 28,98->8); observed above null means people generalize by last digit from few examples more than the model (below = model over-weights last-digit rules on small sets).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    E = pd.DataFrame(ex.tolist()).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    isex = (E == t[:, None]).any(axis=1)
    mn = np.nanmin(E, axis=1); mx = np.nanmax(E, axis=1)
    ld = (np.mod(E, 10) == np.mod(t, 10)[:, None]).any(axis=1)
    m = (n <= 2) & (~isex) & ld
    return float(r[m].mean()) if m.any() else 0.0
