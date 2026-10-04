# name: consensus_agreement_drift
# description: Agreement with the leave-one-out pair majority in each participant's second half of trials minus their first half; the model has fixed per-person sensitivity (null near 0), so observed below null_mean means responses become noisier/less decisive over the session (fatigue), above means they sharpen.
def test_statistic(df):
    a, b = df["sequence_a"].values, df["sequence_b"].values
    c = df["chose_left"].values.astype(float)
    swap = a > b
    key = np.where(swap, b + "|" + a, a + "|" + b)
    y = np.where(swap, 1 - c, c)
    g = pd.DataFrame({"k": key, "y": y})
    s = g.groupby("k")["y"].transform("sum").values
    n = g.groupby("k")["y"].transform("count").values
    loo = (s - y) / np.clip(n - 1, 1, None)
    agree = np.where(loo > 0.5, y, np.where(loo < 0.5, 1 - y, 0.5))
    med = df.groupby("participant_id")["trial_index"].transform("median").values
    late = df["trial_index"].values > med
    return float(agree[late].mean() - agree[~late].mean())
