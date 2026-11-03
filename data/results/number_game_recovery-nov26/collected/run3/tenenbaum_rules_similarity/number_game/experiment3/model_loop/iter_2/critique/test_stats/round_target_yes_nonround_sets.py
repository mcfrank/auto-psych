# name: round_target_yes_nonround_sets
# description: Among sets with no multiple-of-10 example, yes rate for multiple-of-10 targets minus yes rate for other non-example targets within 3 of a multiple of 10 (and not ending in 0); observed above null means people over-accept salient round numbers relative to the model.
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
    anyround = np.any(valid & (mat % 10 == 0), 1)
    base = ~anyround & ~shown
    tm = t % 10
    a = base & (tm == 0)
    b = base & (tm != 0) & ((tm <= 3) | (tm >= 7))
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
