# name: participant_heads_preference_variance
# description: Variance across participants of their rate of choosing the more-heads sequence (pairs differing in #H); observed above null means the heads preference differs between people, which the model's single shared heads weight under-produces.
def test_statistic(df):
    ha = df["sequence_a"].str.count("H").to_numpy(); hb = df["sequence_b"].str.count("H").to_numpy()
    m = ha != hb
    c = df["chose_left"].to_numpy()
    v = np.where(ha > hb, c, 1 - c).astype(float)
    s = pd.Series(v[m]).groupby(df["participant_id"].to_numpy()[m]).mean()
    return float(s.var()) if len(s) > 1 else 0.0
