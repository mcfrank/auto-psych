# name: late_minus_early_majority_agreement
# description: Rate of agreeing with the pair's majority choice in each participant's second half of trials minus that in the first half (by trial_index); observed above null_mean means people become more consistent/decisive over the session (model assumes constant sensitivity), below means fatigue/noise increases.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    lo = np.where(a < b, a, b); key = pd.Series(lo + "|" + np.where(a < b, b, a), index=df.index)
    chose_lo = pd.Series(np.where(a.to_numpy() == lo, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy()).astype(float), index=df.index)
    rate = chose_lo.groupby(key).transform("mean")
    agree = np.where(rate > 0.5, chose_lo, np.where(rate < 0.5, 1 - chose_lo, 0.5))
    med = df.groupby("participant_id")["trial_index"].transform("median")
    late = (df["trial_index"] > med).to_numpy()
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(agree[late].mean() - agree[~late].mean())
