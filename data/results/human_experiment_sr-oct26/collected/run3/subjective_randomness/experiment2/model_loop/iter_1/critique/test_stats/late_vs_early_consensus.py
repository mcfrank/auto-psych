# name: late_vs_early_consensus
# description: Agreement with the item-level majority choice in the second half of each session minus the first half; observed < null means responses become noisier with trial position (fatigue), > null means they sharpen, which the stationary model cannot produce.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    y = df["chose_left"].to_numpy().astype(bool)
    lo = np.where(a < b, a, b); hi = np.where(a < b, b, a)
    chosen = np.where(y, a, b)
    key = pd.Series(lo + "|" + hi)
    chose_lo = pd.Series((chosen == lo).astype(float))
    maj_lo = chose_lo.groupby(key).transform("mean").to_numpy() >= 0.5
    agree = (chose_lo.to_numpy() == maj_lo).astype(float)
    t = df["trial_index"].to_numpy()
    med = df.groupby("participant_id")["trial_index"].transform("median").to_numpy()
    late = t > med
    return float(agree[late].mean() - agree[~late].mean())
