# name: ones_digit_set_tens_digit_leak
# description: Among sets of >=2 examples all sharing a nonzero ones digit d, the yes rate to non-example targets that contain d only in the tens place (e.g. 70-79 after 97,77,87,47); observed above the null means people generalize to the "contains digit d" rule more than the model predicts, below means the model over-weights "contains d".
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    ones = ex.map(lambda e: {int(x) % 10 for x in e})
    n = ex.map(len).to_numpy()
    shared = ones.map(lambda s: next(iter(s)) if len(s) == 1 else -1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    isex = np.array([str(a) in b for a, b in zip(t, ex)])
    m = (n >= 2) & (shared > 0) & (t // 10 == shared) & (t % 10 != shared) & ~isex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
