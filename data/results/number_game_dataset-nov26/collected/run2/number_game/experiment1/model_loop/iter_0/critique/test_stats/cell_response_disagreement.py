# name: cell_response_disagreement
# description: Mean over (set, target) cells of p_hat*(1-p_hat), where p_hat is the cell's yes-rate; observed > null means humans' yes-rates are less extreme (more graded/noisy) than the model's near-0/1 posterior predictions, observed < null means more extreme.
def test_statistic(df):
    p = df.groupby(["number_set", "target"])["rating"].mean()
    return float((p * (1 - p)).mean())
