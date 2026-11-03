# name: per_set_yes_rate_sd
# description: Standard deviation across number_sets of each set's overall yes rate; t_observed above null_mean means people differentiate sets (tight vs broad concepts) more sharply than the model, below means less.
def test_statistic(df):
    g = df.groupby(df["number_set"].astype(str))["rating"].mean()
    if len(g) < 2:
        return 0.0
    return float(g.std())
