# name: rule_set_member_far_outspan
# description: Yes rate for targets >10 outside the example span that are multiples of k (k in 3..11) when all examples (>=3) are multiples of k; observed above null means the model under-generalises a clear arithmetic rule beyond the examples' span.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",")
    lens = s.str.len().values
    t = df["target"].astype(int).values
    mat = np.full((len(df), 4), -1)
    for j in range(4):
        col = s.str[j]
        mat[:, j] = pd.to_numeric(col, errors="coerce").fillna(-1).astype(int).values
    valid = mat > 0
    mn = np.where(valid, mat, 1000).min(1); mx = np.where(valid, mat, -1).max(1)
    out = ((t < mn - 10) | (t > mx + 10))
    hit = np.zeros(len(df), bool)
    for k in [3, 4, 5, 6, 7, 8, 9, 11]:
        allk = np.all(~valid | (mat % k == 0), axis=1)
        hit |= allk & (t % k == 0)
    m = hit & out & (lens >= 3)
    return float(df["rating"].values[m].mean()) if m.any() else 0.0
