# name: pair_yes_rate_dispersion
# description: Standard deviation across (set, target) pairs of the pair-level yes-rate; observed above null means human judgments are more all-or-none (more extreme, more set/target-specific) than the model's graded overlap probabilities, below means the model is too extreme.
def test_statistic(df):
    g = df.groupby([df["number_set"].astype(str), df["target"].astype(int)])["rating"].mean()
    return float(g.std(ddof=0))
