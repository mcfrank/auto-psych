# name: within_set_target_spread
# description: Mean over sets with at least 5 distinct asked targets of the standard deviation, across targets, of each (set, target) cell's yes rate; observed above null_mean means people's generalization profiles within a set are more sharply contrasted (all-or-none across targets) than the model predicts, below means flatter/more graded.
def test_statistic(df):
    cell = df.groupby(["number_set", "target"])["rating"].mean()
    g = cell.groupby(level=0)
    sd = g.std(ddof=0)
    cnt = g.size()
    sd = sd[cnt >= 5]
    if len(sd) == 0:
        return 0.0
    return float(sd.mean())
