# name: uniform_digit_set_same_parity_yes
# description: In sets of >=2 examples that all share one final digit, yes rate for targets of the same parity but a different final digit (not examples); t_observed below null_mean means people apply the shared-final-digit rule more sharply than the model's partial parity credit (model over-produces yes for parity-only matches).
def test_statistic(df):
    t = df["target"].astype(int).values
    sets = df["number_set"].astype(str).values
    sel = np.zeros(len(t), bool)
    for s in np.unique(sets):
        xs = np.array([int(v) for v in s.split(",")])
        if len(xs) < 2 or len(np.unique(xs % 10)) != 1:
            continue
        m = sets == s
        tt = t[m]
        d0 = xs[0] % 10
        sel[m] = ((tt % 2) == (d0 % 2)) & ((tt % 10) != d0)
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[sel].mean())
