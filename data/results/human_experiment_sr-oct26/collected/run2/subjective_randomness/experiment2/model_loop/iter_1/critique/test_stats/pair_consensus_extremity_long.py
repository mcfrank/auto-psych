# name: pair_consensus_extremity_long
# description: Mean over length-8 pairs of |proportion choosing left - 0.5|; observed above null means people agree on long-sequence pairs more strongly than the model predicts (model too noisy/under-confident there), below null means the model is over-confident.
def test_statistic(df):
    m = (df["sequence_a"].str.len() == 8).to_numpy()
    sub = df.loc[m]
    if len(sub) == 0:
        return 0.0
    key = sub["sequence_a"] + "|" + sub["sequence_b"]
    p = sub.groupby(key)["chose_left"].mean()
    return float((p - 0.5).abs().mean())
