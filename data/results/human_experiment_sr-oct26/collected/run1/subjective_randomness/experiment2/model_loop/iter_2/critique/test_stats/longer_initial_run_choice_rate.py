# name: longer_initial_run_choice_rate
# description: Among trials whose sequences open with runs of different length (as a share of the length), the proportion choosing the sequence with the LONGER opening run; observed above the null means people tolerate (or favour) an opening streak more than the model predicts, below means they penalise it more (the model weighs only the final and longest runs).
def test_statistic(df):
    def first_run(s):
        k = 1
        while k < len(s) and s[k] == s[0]:
            k += 1
        return k / len(s)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: first_run(s) for s in u}
    fa = df["sequence_a"].map(m).to_numpy(dtype=float)
    fb = df["sequence_b"].map(m).to_numpy(dtype=float)
    sel = fa != fb
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(fa > fb, c, 1 - c)
    return float(chose[sel].mean())
