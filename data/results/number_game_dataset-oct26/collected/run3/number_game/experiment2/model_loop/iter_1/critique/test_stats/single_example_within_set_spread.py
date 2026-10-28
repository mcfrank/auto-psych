# name: single_example_within_set_spread
# description: Mean over single-example sets of the standard deviation (across targets) of per-(set,target) yes-rates; observed > model means people discriminate between targets after one example more sharply than the model's evidence-scarce guessing allows (guessing too flat), observed < model means responses are flatter than predicted.
def test_statistic(df):
    s = df["number_set"].astype(str)
    sub = df[~s.str.contains(",")]
    if len(sub) == 0:
        return 0.0
    cell = sub.groupby(["number_set", "target"])["rating"].mean().reset_index()
    sd = cell.groupby("number_set")["rating"].std(ddof=0)
    sd = sd[cell.groupby("number_set").size() >= 3]
    if len(sd) == 0:
        return 0.0
    return float(sd.mean())
