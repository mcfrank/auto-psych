# name: longer_initial_run_choice
# description: Among trials whose sequences differ in the length of their OPENING run (identical flips at the start), the proportion choosing the sequence with the longer opening run; observed below null_mean means people penalise opening streaks more than the model (which weights only the final and longest runs) predicts, above means less.
def test_statistic(df):
    def first_run(s):
        c = 1
        while c < len(s) and s[c] == s[0]:
            c += 1
        return c
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    fr = {s: first_run(s) for s in seqs}
    ra = df["sequence_a"].map(fr).to_numpy()
    rb = df["sequence_b"].map(fr).to_numpy()
    y = df["chose_left"].to_numpy()
    m = ra != rb
    if m.sum() == 0:
        return 0.5
    return float(np.where(ra[m] > rb[m], y[m], 1 - y[m]).mean())
