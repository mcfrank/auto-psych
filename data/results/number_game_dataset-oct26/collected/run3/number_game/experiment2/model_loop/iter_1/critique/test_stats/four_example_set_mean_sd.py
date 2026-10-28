# name: four_example_set_mean_sd
# description: Standard deviation, across four-example sets, of each set's overall yes-rate; observed > model means sets differ in how readily people say "yes" more than the model predicts (model too homogeneous across set types), observed < model means less heterogeneity than predicted.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    sub = df[n == 4]
    if len(sub) == 0:
        return 0.0
    g = sub.groupby("number_set")["rating"].mean()
    return float(g.std(ddof=0))
