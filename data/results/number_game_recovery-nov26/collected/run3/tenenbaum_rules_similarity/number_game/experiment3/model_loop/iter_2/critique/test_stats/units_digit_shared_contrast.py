# name: units_digit_shared_contrast
# description: Among sets with >=2 examples whose examples do not all share a last digit, yes rate for non-example targets sharing a last digit with some example minus yes rate for targets sharing none; observed above null means people use loose last-digit similarity the model lacks.
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
    u = np.where(valid, mat % 10, -1)
    same_end = np.all(~valid | (u == u[:, :1]), 1)
    share = np.any(valid & (u == (t % 10)[:, None]), 1)
    base = (n >= 2) & ~same_end & ~shown
    a = base & share; b = base & ~share
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
