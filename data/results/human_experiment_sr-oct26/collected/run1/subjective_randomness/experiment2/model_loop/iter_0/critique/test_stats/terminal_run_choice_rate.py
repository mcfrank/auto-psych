# name: terminal_run_choice_rate
# description: Among pairs whose sequences differ in the length of the run at their end (final streak), the proportion of choices of the sequence with the longer final run; observed < null means people penalise streaks at the end of a sequence (recency) more than the model's position-blind longest-run term predicts.
def test_statistic(df):
    def trun(s):
        s = str(s); k = 1
        while k < len(s) and s[-k-1] == s[-1]:
            k += 1
        return k
    ua = df["sequence_a"].map({u: trun(u) for u in df["sequence_a"].unique()})
    ub = df["sequence_b"].map({u: trun(u) for u in df["sequence_b"].unique()})
    m = (ua != ub).values
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].values[m]
    left_longer = (ua > ub).values[m]
    return float(np.mean(np.where(left_longer, c, 1 - c)))
