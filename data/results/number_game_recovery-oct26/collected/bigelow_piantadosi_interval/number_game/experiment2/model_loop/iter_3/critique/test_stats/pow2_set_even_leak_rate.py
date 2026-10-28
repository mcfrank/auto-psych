# name: pow2_set_even_leak_rate
# description: Among sets whose examples are all powers of 2 (2..64), the yes rate to even non-example targets that are NOT powers of 2; observed above the null means people still leak to "even numbers" more than the size-weighted rule prior predicts (model over-commits to powers of 2), below means the model over-generalizes to evens.
def test_statistic(df):
    pw = {2, 4, 8, 16, 32, 64}
    ex = df["number_set"].astype(str).str.split(",")
    allpw = ex.map(lambda e: all(int(x) in pw for x in e)).to_numpy()
    t = df["target"].astype(int).to_numpy()
    isex = np.array([str(a) in b for a, b in zip(t, ex)])
    m = allpw & (t % 2 == 0) & ~np.isin(t, list(pw)) & ~isex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
