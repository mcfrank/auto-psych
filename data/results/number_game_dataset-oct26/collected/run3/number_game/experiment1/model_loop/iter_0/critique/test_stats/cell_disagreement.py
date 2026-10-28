# name: cell_disagreement
# description: Mean over (set, target) cells of p_hat*(1-p_hat), where p_hat is the cell's yes rate; observed below null_mean means humans agree more (more extreme, all-or-none generalization) than the model's graded posterior predicts, above means humans are more split.
def test_statistic(df):
    p = df.groupby(["number_set", "target"])["rating"].mean()
    return float((p * (1 - p)).mean())
