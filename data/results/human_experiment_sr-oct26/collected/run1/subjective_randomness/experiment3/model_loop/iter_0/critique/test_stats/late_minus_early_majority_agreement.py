# name: late_minus_early_majority_agreement
# description: Agreement with each pair's population-majority choice in the last third of a session's trials minus that in the first third; observed below null_mean means choices get noisier with trial position (fatigue) beyond the model's constant lapse, above means they get sharper.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    canon_left = a <= b
    key = np.where(canon_left, a, b) + "|" + np.where(canon_left, b, a)
    chose_canon = np.where(canon_left, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    s = pd.Series(chose_canon, index=key)
    rate = s.groupby(level=0).mean()
    pref = (rate.reindex(key).to_numpy() >= 0.5).astype(int)
    agree = (chose_canon == pref).astype(float)
    ti = df["trial_index"].to_numpy()
    rank = df.groupby("participant_id")["trial_index"].rank(pct=True).to_numpy()
    early = agree[rank <= 1/3]; late = agree[rank > 2/3]
    return float(late.mean() - early.mean())
