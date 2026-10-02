# name: alternation_choice_perseveration
# description: Among consecutive trial pairs of a participant where both trials' sequences differ in switch count, P(choose the more-switching sequence | did so on the previous trial) minus P(choose it | did not); observed above null_mean means people's alternation preference drifts/perseverates across trials (sequential dependence) that the model's trial-independent choices cannot produce.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    sa = d["sequence_a"].str.count("HT").to_numpy() + d["sequence_a"].str.count("TH").to_numpy()
    sb = d["sequence_b"].str.count("HT").to_numpy() + d["sequence_b"].str.count("TH").to_numpy()
    cl = d["chose_left"].to_numpy()
    valid = sa != sb
    more = np.where(sa > sb, cl == 1, cl == 0).astype(float)
    pid = d["participant_id"].to_numpy()
    same = pid[1:] == pid[:-1]
    ok = same & valid[1:] & valid[:-1]
    prev = more[:-1][ok]; cur = more[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
