# name: ones_digit_set_unrelated_target_rate
# description: Among sets of >=2 examples all sharing a ones digit d (incl. multiples of 10), the yes rate to targets that neither end in d nor contain d at all; observed above the null means people's yes responses leak beyond the digit rule (via ranges/parity) more than the model predicts, below means the model leaks too much.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    ones = ex.map(lambda e: {int(x) % 10 for x in e})
    n = ex.map(len).to_numpy()
    shared = ones.map(lambda s: next(iter(s)) if len(s) == 1 else -1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    tens = t // 10
    m = (n >= 2) & (shared >= 0) & (t % 10 != shared) & ((tens != shared) | (shared == 0)) & (t != 100)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
