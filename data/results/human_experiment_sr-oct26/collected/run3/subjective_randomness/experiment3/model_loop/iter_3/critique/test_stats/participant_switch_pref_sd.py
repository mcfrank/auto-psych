# name: participant_switch_pref_sd
# description: Across participants, the standard deviation of each person's rate of choosing the more-switching sequence (pairs differing in switch count); observed above null means people differ in their switching preference more than the model's person-level switch-belief and sensitivity spread produces, below means less.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    ka = a.str.count("HT") + a.str.count("TH"); kb = b.str.count("HT") + b.str.count("TH")
    m = (ka != kb).to_numpy()
    if m.sum() == 0:
        return 0.0
    y = df["chose_left"].to_numpy()[m]
    left_more = (ka > kb).to_numpy()[m]
    c = np.where(left_more, y, 1 - y)
    r = pd.Series(c).groupby(df["participant_id"].to_numpy()[m]).mean()
    return float(r.std(ddof=0)) if len(r) > 1 else 0.0
