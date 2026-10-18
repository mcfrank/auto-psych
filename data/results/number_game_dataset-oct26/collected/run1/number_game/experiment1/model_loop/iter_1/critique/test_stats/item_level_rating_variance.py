# name: item_level_rating_variance
# description: Variance across (set, target) items of the item mean yes rate; observed above null means people's judgments are more sharply all-or-none across items than the model's graded similarity predicts (model too flat), below means the model over-discriminates.
def test_statistic(df):
    m = df.groupby(["number_set", "target"])["rating"].mean()
    return float(m.var(ddof=0))
