# name: within_set_response_spread
# description: Mean (over sets with >=10 trials) of the within-set variance of yes/no responses p(1-p); observed above the model's null means people discriminate targets within a set less sharply (more uniform/graded) than predicted, below means they are more all-or-none per set than the model.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    if len(g) == 0:
        return 0.0
    return float((g["mean"] * (1 - g["mean"])).mean())
