# name: within_cell_response_variability
# description: Mean over (set, target) cells of p(1-p), p = cell yes rate across repeated presentations; observed above null means the model's predicted probabilities are too extreme (near 0/1) relative to graded human responding, below means too graded.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].mean()
    return float((g * (1 - g)).mean())
