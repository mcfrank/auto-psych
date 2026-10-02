# name: majority_agreement_trial_slope
# description: Difference in agreement with each pair's observed majority choice between the second and first half of trial_index (late minus early); observed < null means responses become noisier/less consistent over the session (fatigue), which the stationary model cannot produce.
def test_statistic(df):
    a = df["sequence_a"].to_numpy()
    b = df["sequence_b"].to_numpy()
    y = df["chose_left"].to_numpy()
    first = np.where(a < b, a, b)
    second = np.where(a < b, b, a)
    chose_first = np.where(a < b, y, 1 - y)
    key = pd.Series(first) + "|" + pd.Series(second)
    item_rate = pd.Series(chose_first).groupby(key).transform("mean").to_numpy()
    maj = (item_rate >= 0.5).astype(float)
    agree = (chose_first == maj).astype(float)
    t = df["trial_index"].to_numpy()
    late = t > np.median(t)
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(agree[late].mean() - agree[~late].mean())
