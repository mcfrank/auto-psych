# name: two_example_example_target_rate
# description: For two-example sets, yes rate when the target is one of the shown examples (59,14->59 .44; 3,7->3 .33); observed below null means the model over-endorses the examples themselves (people withhold yes ~40% of the time, a lapse/doubt the model lacks), above = model under-endorses them.
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    sub = df[n == 2]
    parts = sub["number_set"].str.split(",", expand=True).astype(int)
    t = sub["target"].astype(int)
    m = (t == parts[0]) | (t == parts[1])
    if m.sum() == 0:
        return 0.0
    return float(sub["rating"][m].mean())
