# name: within_set_target_spread
# description: Mean, over example sets asked about with >= 8 distinct targets, of the variance across that set's targets of the per-(set,target) yes-rate; observed above model means people differentiate targets within a set more sharply than the model's (too flat) posterior-predictive probabilities, below means the model differentiates targets more than people do.
def test_statistic(df):
    cell = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    stats = cell.groupby("number_set")["rating"].agg(["var", "size"])
    stats = stats[stats["size"] >= 8]
    if len(stats) == 0:
        return 0.0
    return float(stats["var"].mean())
