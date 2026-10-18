# name: within_set_target_spread
# description: Mean over sets of the std of target-level yes rates within the same set (how differently targets for one set are treated); observed above null means the model under-differentiates targets within a set (too graded), below means it over-differentiates.
def test_statistic(df):
    m = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    sd = m.groupby("number_set")["rating"].std(ddof=0)
    cnt = m.groupby("number_set")["rating"].size()
    sd = sd[cnt >= 2]
    if len(sd) == 0:
        return 0.0
    return float(sd.mean())
