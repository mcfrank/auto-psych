# name: late_vs_early_discrimination
# description: Correlation between each answer and its (set,target) cell's mean yes rate in the second half of each session (trial_index>=225) minus the same in the first half; positive vs the null means people get sharper/more consistent with the consensus as the session goes on (negative: noisier), which a static model cannot produce.
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    y = df["rating"].astype(float)
    cm = y.groupby(key).transform("mean").values
    yv = y.values
    late = df["trial_index"].astype(int).values >= 225
    def corr(m):
        a, b = yv[m], cm[m]
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    return corr(late) - corr(~late)
