# name: alternation_pref_drift_over_trials
# description: Rate of choosing the more-alternating sequence in the second half of each session minus the first half; positive observed vs null means preferences drift toward alternation over the session, which the static model cannot produce.
def test_statistic(df):
    def alts(s):
        return s.map({q: sum(q[i] != q[i - 1] for i in range(1, len(q))) for q in s.unique()}).to_numpy()
    aa = alts(df["sequence_a"]); ab = alts(df["sequence_b"])
    y = df["chose_left"].to_numpy()
    c = np.where(aa > ab, y, 1 - y)
    t = df["trial_index"].to_numpy()
    med = pd.Series(t).groupby(df["participant_id"].to_numpy()).transform("median").to_numpy()
    m = aa != ab
    late = m & (t > med); early = m & (t <= med)
    if late.sum() == 0 or early.sum() == 0:
        return 0.0
    return float(c[late].mean() - c[early].mean())
