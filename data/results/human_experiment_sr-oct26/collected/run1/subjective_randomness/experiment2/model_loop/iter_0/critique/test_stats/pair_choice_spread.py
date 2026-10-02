# name: pair_choice_spread
# description: Mean absolute deviation from 0.5 of the per-pair proportion choosing the alphabetically-first sequence (orientation-normalised, pooled over participants); observed > null means stimulus-level preferences are more extreme/consistent than the model predicts (it is too noisy/lapse-heavy), observed < null means the model is overconfident.
def test_statistic(df):
    a = df["sequence_a"].values.astype(str)
    b = df["sequence_b"].values.astype(str)
    first_is_a = a <= b
    key = np.where(first_is_a, a + "|" + b, b + "|" + a)
    chose_first = np.where(first_is_a, df["chose_left"].values, 1 - df["chose_left"].values)
    p = pd.Series(chose_first).groupby(key).mean()
    return float(np.mean(np.abs(p.values - 0.5)))
