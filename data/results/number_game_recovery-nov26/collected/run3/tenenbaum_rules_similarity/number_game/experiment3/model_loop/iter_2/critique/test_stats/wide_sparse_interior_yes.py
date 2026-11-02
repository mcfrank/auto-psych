# name: wide_sparse_interior_yes
# description: For sets of >=3 examples spanning >=30, yes rate for non-example targets strictly inside the span; observed below null means people do not fill the gaps of wide sparse sets as the model's ranges/rules imply (above: they fill gaps more).
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",")
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    mat = np.full((len(df), 4), -1)
    for j in range(4):
        mat[:, j] = pd.to_numeric(s.str[j], errors="coerce").fillna(-1).astype(int).values
    valid = mat > 0
    n = valid.sum(1)
    shown = np.any(valid & (mat == t[:, None]), 1)
    lo = np.where(valid, mat, 999).min(1)
    hi = np.where(valid, mat, -1).max(1)
    m = (n >= 3) & ((hi - lo) >= 30) & ~shown & (t > lo) & (t < hi)
    return float(r[m].mean()) if m.any() else 0.0
