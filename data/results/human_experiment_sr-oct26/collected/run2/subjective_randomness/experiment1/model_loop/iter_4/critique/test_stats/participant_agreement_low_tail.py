# name: participant_agreement_low_tail
# description: 10th percentile across participants of each person's agreement rate with the leave-one-out majority choice on each pair; observed below null_mean means there are near-random or contrarian responders beyond what the log-normal sensitivity distribution produces.
def test_statistic(df):
    a, b = df["sequence_a"].values, df["sequence_b"].values
    c = df["chose_left"].values.astype(float)
    swap = a > b
    key = np.where(swap, b + "|" + a, a + "|" + b)
    chose_first = np.where(swap, 1 - c, c)
    g = pd.DataFrame({"k": key, "y": chose_first, "p": df["participant_id"].values})
    s = g.groupby("k")["y"].transform("sum")
    n = g.groupby("k")["y"].transform("count")
    loo = (s - g["y"]) / (n - 1).clip(lower=1)
    agree = np.where(loo > 0.5, g["y"], np.where(loo < 0.5, 1 - g["y"], 0.5))
    per = pd.Series(agree).groupby(g["p"].values).mean()
    return float(np.percentile(per.values, 10))
