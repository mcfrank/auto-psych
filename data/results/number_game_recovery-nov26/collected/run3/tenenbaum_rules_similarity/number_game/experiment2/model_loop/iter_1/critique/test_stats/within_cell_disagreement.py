# name: within_cell_disagreement
# description: Over (number_set, target) cells answered by >=2 people, the mean probability that two different responses in the cell disagree (unbiased pairwise disagreement 2k(n-k)/(n(n-1))); observed < null means people are more consistent (p_yes more extreme) than the model's predictions, observed > null means the model's predictions are too confident relative to human disagreement.
def test_statistic(df):
    g = df.groupby([df["number_set"].astype(str), df["target"]])["rating"].agg(["sum", "count"])
    g = g[g["count"] >= 2]
    k, n = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    return float(np.mean(2 * k * (n - k) / (n * (n - 1)))) if len(g) else 0.0
