# name: odd_set_even_target_yes
# description: Yes rate for even targets inside the example span for all-odd sets of >=3 examples; observed above null means the model rejects parity-violating in-range targets more than people do.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",")
    t = df["target"].astype(int).values
    mat = np.full((len(df), 4), -1)
    for j in range(4):
        mat[:, j] = pd.to_numeric(s.str[j], errors="coerce").fillna(-1).astype(int).values
    valid = mat > 0
    n = valid.sum(1)
    allodd = np.all(~valid | (mat % 2 == 1), 1)
    mn = np.where(valid, mat, 1000).min(1); mx = np.where(valid, mat, -1).max(1)
    m = allodd & (n >= 3) & (t % 2 == 0) & (t > mn) & (t < mx)
    return float(df["rating"].values[m].mean()) if m.any() else 0.0
