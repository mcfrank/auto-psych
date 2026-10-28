# name: cell_response_graded_ness_multi
# description: Over (set, target) cells from sets of 2+ examples, the mean of p*(1-p) where p is the cell's yes-rate across its repeated responses; observed < model means people's answers per item are more consensual/all-or-none than the model's graded posterior-mixture probabilities produce, observed > model means more within-item disagreement than predicted.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    d = df.loc[n >= 2]
    p = d.groupby(["number_set", "target"])["rating"].mean().to_numpy(dtype=float)
    return float(np.mean(p * (1 - p))) if len(p) else 0.0
