# name: cell_response_dispersion
# description: Mean over (number_set, target) cells of the within-cell Bernoulli variance p(1-p) of the repeated responses; observed below the null means people agree with each other on each item more (responses more deterministic/extreme) than the model's predicted probabilities imply, above means the model is overconfident.
def test_statistic(df):
    p = df.groupby(["number_set", "target"])["rating"].mean().astype(float)
    return float((p * (1 - p)).mean())
