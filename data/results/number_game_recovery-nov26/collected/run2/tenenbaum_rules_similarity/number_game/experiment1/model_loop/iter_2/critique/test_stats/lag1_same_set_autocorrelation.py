# name: lag1_same_set_autocorrelation
# description: Pearson correlation between a trial's rating and the previous row's rating, over consecutive rows (trial_index increases by 1) with the same number_set; observed above null means sequential response perseveration/carry-over that the trial-independent model cannot produce, below null means alternation.
def test_statistic(df):
    r = df["rating"].to_numpy().astype(float)
    ti = df["trial_index"].to_numpy()
    s = df["number_set"].astype(str).to_numpy()
    if len(r) < 3:
        return 0.0
    m = (ti[1:] - ti[:-1] == 1) & (s[1:] == s[:-1])
    a, b = r[1:][m], r[:-1][m]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
