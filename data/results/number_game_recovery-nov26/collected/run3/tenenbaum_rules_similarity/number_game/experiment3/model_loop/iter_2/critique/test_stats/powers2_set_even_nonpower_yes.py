# name: powers2_set_even_nonpower_yes
# description: For sets of >=2 examples that are all powers of 2, yes rate for even non-power-of-2 targets; observed above null means people broaden power-of-2 sets toward 'even numbers' more than the model does (below: they hold the narrow power rule more tightly).
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
    pw = np.isin(mat, [2, 4, 8, 16, 32, 64])
    allp = np.all(~valid | pw, 1) & (n >= 2)
    m = allp & (t % 2 == 0) & ~np.isin(t, [2, 4, 8, 16, 32, 64]) & ~shown
    return float(r[m].mean()) if m.any() else 0.0
