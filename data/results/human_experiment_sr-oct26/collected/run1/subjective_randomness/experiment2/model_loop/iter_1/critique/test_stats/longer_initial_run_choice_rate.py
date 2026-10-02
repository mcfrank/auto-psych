# name: longer_initial_run_choice_rate
# description: Among trials whose sequences begin with runs of different length, the proportion choosing the sequence with the LONGER initial run; observed below the null means people penalise an opening streak (primacy) more than the model, which weighs the final run but not the first, predicts.
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
