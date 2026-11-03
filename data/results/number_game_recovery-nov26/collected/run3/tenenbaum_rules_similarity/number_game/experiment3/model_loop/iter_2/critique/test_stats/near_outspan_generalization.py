# name: near_outspan_generalization
# description: Yes rate for non-example targets lying 1-5 outside the [min,max] span of sets with >=2 examples; observed above null means people generalize just beyond the examples more than the model's interval/rule posterior predicts (below: they stop sharper at the edges).
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
    d = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0))
    m = (n >= 2) & ~shown & (d >= 1) & (d <= 5)
    return float(r[m].mean()) if m.any() else 0.0
