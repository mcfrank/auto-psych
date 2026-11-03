# name: small_vs_large_target_contrast
# description: Yes rate for non-example targets <=10 minus yes rate for non-example targets >=91, pooled over all sets; observed above null means people favour small numbers as members regardless of the set more than the model predicts (below: favour large numbers).
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
    a = ~shown & (t <= 10); b = ~shown & (t >= 91)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
